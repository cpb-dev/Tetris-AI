from datetime import datetime
import io
from wsgiref.validate import IteratorWrapper
import pygame
import pickle
import numpy as np
import torch
import logging
import sys

from tetris import main, tetrisGame, findPiece
from n_network.algor import get_score, Population
from n_network.boardInfo import check_turns, make_move, press_btn, \
    check_needed_moves, feature_names
from multiprocessing import Pool, cpu_count

#save and load states, inspired by pyboy
#def save_state(self, file_like_object):
#    if isinstance(file_like_object, str):
#        raise Exception("String not allowed. Did you specify a filepath instead of a file-like object?")
#    self.mb.save_state(IteratorWrapper(file_like_object))
#def load_state(self, file_like_object):
#    if isinstance(file_like_object, str):
#        raise Exception("String not allowed. Did you specify a filepath instead of a file-like object?")
#    self.mb.load_state(IteratorWrapper(file_like_object))

logger = logging.getLogger("tetris")
logger.setLevel(logging.INFO)

fh = logging.FileHandler('logs.out')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

#Network config
max_fitness = 0
child_runs = 1
max_score = 1000 # Can change

pop_size = 25
epochs = 50
population = None
n_workers = cpu_count()

def eval_network(epoch, child_index, child_model):
    tetris = tetrisGame()
    #tetrisGame()

    run = 0
    scores = []
    levels =[]
    lines = []

    while run < child_runs:
        best_action_score = np.NINF
        best_action = {'Turn' : 0, 'Left': 0, 'Right': 0}
        begin_state = io.BytesIO()
        begin_state.seek(0)
        #save_state(begin_state)

        block_tile = findPiece(main.current_piece) #main.c_piece()
        print(block_tile)
        #getattr(main, 'piece_letter')
        turns = check_turns(block_tile)
        lefts, rights = check_needed_moves(block_tile)

        for move_dir in make_move('Middle', tetris, n_dir = 1, n_turn = turns):
            score = get_score(tetris, child_model)
            if score is not None and score >= best_action_score:
                best_action_score = score
                best_action = {'Turn': move_dir['Turn'], 
                               'Left': move_dir['Left'], 
                               'Right': move_dir['Right']
                                }
            begin_state.seek(0)
            #load_state(begin_state)
        
        for move_dir in make_move('Left', tetris, n_dir = lefts, n_turn = turns):
            score = get_score(tetris, child_model)
            if score is not None and score >= best_action_score:
                best_action_score = score
                best_action = {'Turn': move_dir['Turn'], 
                               'Left': move_dir['Left'], 
                               'Right': move_dir['Right']
                               }
            begin_state.seek(0)
            #load_state(begin_state)

        for move_dir in make_move('Right', tetris, n_dir = rights, n_turn = turns):
            score = get_score(tetris, child_model)
            if score is not None and score >= best_action_score:
                best_action_score = score
                best_action = {'Turn': move_dir['Turn'], 
                               'Left': move_dir['Left'], 
                               'Right': move_dir['Right']
                               }
            begin_state.seek(0)
            #load_state(begin_state)

        for _ in range (best_action['Turn']):
            press_btn('Up')
        for _ in range (best_action['Left']):
            press_btn('Down')
        for _ in range(best_action['Right']):
            press_btn('Right')
        press_btn('Space')

        if main.run == 'False' or main.score == max_score:
            scores.append(main.score)
            if run == child_runs - 1:
                tetris.close()
            else:
                tetris.reset()
            run += 1

    child_fitness = np.average(scores)
    logger.info("-" * 20)
    logger.info("Iteration %s - child %s" % (epoch, child_index))
    logger.info("Fitness %s" % child_fitness)
    logger.info("Output weight:")
    weights = {}
    for i, j in zip(feature_names, child_model.output.weight.data.tolist()[0]):
        weights[i] = np.round(j, 3)
    logger.info(weights)

    return child_fitness

if __name__ == '__main__':
    e = 0
    p = Pool(n_workers)

    while e < epochs:
        start_time = datetime.now()
        if population is None:
            if e == 0:
                population = Population(size = pop_size)
            else:
                with open('checkpoint/checkpoint-%s.pkl' % (e - 1), 'rb') as f:
                    population = pickle.load(f)
        else:
            population = Population(size=pop_size, old_population=population)

        result = [0] * pop_size
        for i in range(pop_size):
            result[i] = p.apply_async(
                eval_network, (e, i, population.models[i]))

        for i in range(pop_size):
            population.fitnesses[i] = result[i].get()

        logger.info("-" * 20)
        logger.info("Iteration %s fitnesses %s" % (
            e, np.round(population.fitnesses, 2)))
        logger.info(
            "Iteration %s max fitness %s " % (e, np.max(population.fitnesses)))
        logger.info(
            "Iteration %s mean fitness %s " % (e, np.mean(
                population.fitnesses)))
        logger.info("Time took %s" % (datetime.now() - start_time))
        logger.info("Best child output weights:")
        weights = {}
        for i, j in zip(feature_names, population.models[np.argmax(
                population.fitnesses)].output.weight.data.tolist()[0]):
            weights[i] = np.round(j, 3)
        logger.info(weights)
        # Saving population
        with open('checkpoint/checkpoint-%s.pkl' % e, 'wb') as f:
            pickle.dump(population, f)

        if np.max(population.fitnesses) >= max_fitness:
            max_fitness = np.max(population.fitnesses)
            file_name = datetime.strftime(datetime.now(), '%d_%H_%M_') + str(
                np.round(max_fitness, 2))
            # Saving best model
            torch.save(
                population.models[np.argmax(
                    population.fitnesses)].state_dict(),
                'models/%s' % file_name)
        e += 1

    p.join()
    p.close()



