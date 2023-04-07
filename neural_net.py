import time
from multiprocessing import Pool
from typing import ForwardRef
import random
import copy
from itertools import combinations
import json
import numpy as np

from game_board import Connect4Board, Player


def rand(x) -> float:
    """
    Outputs uniform random num between -x and x.
    """
    return x* 2*(random.random()-0.5)


class InputNeuron():
    def __init__(self, weights=None, b=None):
        self.weights = [rand(x=1)] if weights is None else weights
        self.b = rand(x=1) if b is None else b
        self.value = None


Neuron = ForwardRef('Neuron')
class Neuron():
    def __init__(self, children: list[Neuron] | list[InputNeuron] = None, weights=None, b=None):
        self.children = children if children is None else children
        weights = [rand(x=1) for _ in range(len(children))] if weights is None else weights
        self.weights = np.array(weights)
        self.b = rand(x=1) if b is None else b


NeuralNet = ForwardRef('NeuralNet')

class NeuralNet():
    def __init__(self, input_neurons=None, hidden_neurons=None, output_neurons=None):
        self.input_neurons = [InputNeuron() for _ in range(42)] if input_neurons is None else input_neurons
        self.hidden_neurons = [Neuron(children=self.input_neurons) for _ in range(35)] if hidden_neurons is None else hidden_neurons
        self.output_neurons = [Neuron(children=self.hidden_neurons) for _ in range(7)] if output_neurons is None else output_neurons


    def select_move(self, board: Connect4Board) -> int:
        available_moves = board.available_moves()

        X = board.board.flatten()
        assert len(X) == 42
        
        for x, input_neuron in zip(X, self.input_neurons):
            input_neuron.value = x

        weights = np.array([n.weights[0] for n in self.input_neurons])
        values = np.array([n.value for n in self.input_neurons])
        bs = np.array([n.b for n in self.input_neurons])
        input_layer = [max(0,x) for x in weights*values + bs]

        middle_layer = [
            max((n.weights*input_layer).sum() + n.b, 0) for n in self.hidden_neurons
        ]

        outputs = [
            max((n.weights*middle_layer).sum() + n.b, 0) for n in self.output_neurons
        ]
        outputs = [(ind, x) for ind, x in enumerate(outputs)]

        outputs = [x for x in outputs if x[1] > 0 and x[0] in available_moves]
        outputs = sorted(outputs, key=lambda o: o[1], reverse=True)

        if outputs:
            return outputs[outputs.index(max(outputs))][0]
        else:
            return random.choice(available_moves)



    def mutate(self):
        for neuron in self.input_neurons + self.hidden_neurons + self.output_neurons:
            for n in range(len(neuron.weights)):
                if random.random() <= 0.05:
                    neuron.weights[n] = rand(x=1)

            if random.random() <= 0.05:
                neuron.b = rand(x=1)


    def breed(self, other: NeuralNet) -> NeuralNet:
        spawn = copy.deepcopy(self)

        my_neurons = spawn.input_neurons + spawn.hidden_neurons + spawn.output_neurons
        other_neurons = other.input_neurons + other.hidden_neurons + other.output_neurons
        for my_neuron, other_neuron in zip(my_neurons, other_neurons):
            for n in range(len(my_neuron.weights)):
                if random.random() <= 0.5:
                    my_neuron.weights[n] = other_neuron.weights[n]

            if random.random() <= 0.5:
                my_neuron.b = other_neuron.b
        
        return spawn


    def to_json(self):
        input_neurons = []
        for neuron in self.input_neurons:
            input_neurons.append(
                {
                    'weights': list(neuron.weights),
                    'b': neuron.b
                }
            )

        hidden_neurons = []
        for neuron in self.hidden_neurons:
            hidden_neurons.append(
                {
                    'weights': list(neuron.weights),
                    'b': neuron.b
                }
            )

        output_neurons = []
        for neuron in self.output_neurons:
            output_neurons.append(
                {
                    'weights': list(neuron.weights),
                    'b': neuron.b
                }
            )

        d = {
            'input_neurons': input_neurons,
            'hidden_neurons': hidden_neurons,
            'output_neurons': output_neurons,
        }

        return json.dumps(d)


    @staticmethod
    def from_json(filepath: str) -> NeuralNet:
        d = json.loads(open(filepath).read())

        input_neurons = []
        for neuron in d['input_neurons']:
            input_neurons.append(
                InputNeuron(weights=np.array(neuron['weights']), b=neuron['b'])
            )

        hidden_neurons = []
        for neuron in d['hidden_neurons']:
            hidden_neurons.append(
                Neuron(weights=np.array(neuron['weights']), b=neuron['b'], children=input_neurons)
            )

        output_neurons = []
        for neuron in d['output_neurons']:
            output_neurons.append(
                Neuron(weights=np.array(neuron['weights']), b=neuron['b'], children=hidden_neurons)
            )

        return NeuralNet(input_neurons=input_neurons, hidden_neurons=hidden_neurons, output_neurons=output_neurons)


class Players():
    def __init__(self, player_x: NeuralNet, player_o: NeuralNet):
        self.player_x = player_x
        self.player_o = player_o

def play_connect4(players: Players) -> Player:
    player_x = players.player_x
    player_o = players.player_o

    board = Connect4Board()
    
    while board.winner is None:
        if board.whose_turn == Player.X:
            col = player_x.select_move(board=board)
        else:
            col = player_o.select_move(board=board)
        board.drop_piece(col)

    return board.winner 


class Citizen():
    def __init__(self, nn: NeuralNet):
        self.nn = nn
        self.score = 0

    def reset_score(self):
        self.score = 0


def _make_new_generation(citizens: list[Citizen], HALF_choose_2, HALF) -> list[Citizen]:
    # breed top half to replace bottom half
    citizens = sorted(citizens, key=lambda x:x.score, reverse=True)

    random.shuffle(HALF_choose_2)
    choices = HALF_choose_2[:HALF]
    for ind, choice in enumerate(choices):
        m,n = choice
        good1 = citizens[m].nn
        good2 = citizens[n].nn

        spawn = good1.breed(good2)
        citizens[HALF+ind] = Citizen(spawn)

    # mutate each citizen except for king
    for x in citizens[1:]:
        x.nn.mutate()

    # reset scores
    for x in citizens:
        x.reset_score()
    
    return citizens


def genetic_algo():
    N = 30
    HALF = N//2
    HALF_choose_2 = list(combinations(list(range(HALF)), 2))

    xs = [Citizen(NeuralNet()) for _ in range(N)]
    os = [Citizen(NeuralNet()) for _ in range(N)]
    
    # generation
    generation = 0
    while True:
        print('\ngeneration', generation := generation + 1)

        begin = time.time()
        for x in range(N):

            x_player = xs[x]
            players = [Players(player_x=x_player.nn,player_o=o.nn) for o in os]

            # with Pool(3) as p:
            #     winners = p.map(play_connect4, players)

            winners = [play_connect4(player) for player in players]

            for ind, winner in enumerate(winners):
                o_player = os[ind]

                if winner == Player.X:
                    x_player.score += 1
                elif winner == Player.O:
                    o_player.score += 1

        end = time.time()
        print(f'all games: {end-begin}, {len(winners)}, {N}')


        print(f'x_scores: {[x.score for x in xs]}\no_scores: {[o.score for o in os]}')

        # make next generation

        # breed top half to replace bottom half
        xs = _make_new_generation(xs, HALF_choose_2, HALF)
        os = _make_new_generation(os, HALF_choose_2, HALF)

        # save each king to file
        with open('./player-X.json', 'w') as f:
            text = xs[0].nn.to_json()
            f.write(text)

        with open('./player-O.json', 'w') as f:
            text = os[0].nn.to_json()
            f.write(text)



if __name__ == '__main__':
    genetic_algo()
