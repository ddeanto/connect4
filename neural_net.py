from typing import ForwardRef
import random
import copy
from itertools import combinations
import json

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

    def activate(self) -> float:
        m = self.weights[0]
        x = self.value
        b = self.b
        return(
            max(m*x + b, 0)
        ) 


Neuron = ForwardRef('Neuron')

class Neuron():
    def __init__(self, children: list[Neuron] | list[InputNeuron] = None, weights=None, b=None):
        self.children = children if children is None else children
        self.weights = [rand(x=1) for _ in range(len(children))] if weights is None else weights
        self.b = rand(x=1) if b is None else b

    def activate(self) -> float:
        cumsum = self.b
        for weight, child in zip(self.weights, self.children):
            cumsum += weight*child.activate()
        return max(cumsum, 0)


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

        outputs = [(n, neuron.activate()) for n, neuron in enumerate(self.output_neurons)]
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
                    'weights': neuron.weights,
                    'b': neuron.b
                }
            )

        hidden_neurons = []
        for neuron in self.hidden_neurons:
            hidden_neurons.append(
                {
                    'weights': neuron.weights,
                    'b': neuron.b
                }
            )

        output_neurons = []
        for neuron in self.output_neurons:
            output_neurons.append(
                {
                    'weights': neuron.weights,
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
                InputNeuron(weights=neuron['weights'], b=neuron['b'])
            )

        hidden_neurons = []
        for neuron in d['hidden_neurons']:
            hidden_neurons.append(
                Neuron(weights=neuron['weights'], b=neuron['b'], children=input_neurons)
            )

        output_neurons = []
        for neuron in d['output_neurons']:
            output_neurons.append(
                Neuron(weights=neuron['weights'], b=neuron['b'], children=hidden_neurons)
            )

        return NeuralNet(input_neurons=input_neurons, hidden_neurons=hidden_neurons, output_neurons=output_neurons)


def play_connect4(playerX: NeuralNet, playerY: NeuralNet) -> Player:
    board = Connect4Board()
    
    while board.winner is None:
        if board.whose_turn == Player.X:
            col = playerX.select_move(board=board)
        else:
            col = playerY.select_move(board=board)
        board.drop_piece(col)

    # print(f'winner: {Player(board.winner)}\n{board}')
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
    N = 40
    HALF = N//2
    HALF_choose_2 = list(combinations(list(range(HALF)), 2))

    xs = [Citizen(NeuralNet()) for _ in range(N)]
    os = [Citizen(NeuralNet()) for _ in range(N)]
    
    # generation
    generation = 0
    while True:
        print('generation', generation := generation + 1)

        count = 0
        for x in range(N):
            for o in range(N):
                count += 1
                if count%100 == 0:
                    print(count)

                x_player = xs[x]
                o_player = os[o]

                winner = play_connect4(x_player.nn, o_player.nn)
                if winner == Player.X:
                    x_player.score += 1
                elif winner == Player.O:
                    o_player.score += 1

        print(f'\ngeneration: {generation}\nx_scores: {[x.score for x in xs]}\no_scores: {[o.score for o in os]}')

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
    # genetic_algo()


    nn = NeuralNet()
    with open('./player-TEST.json', 'w') as f:
        text = nn.to_json()
        f.write(text)

    nn2 = NeuralNet.from_json('./player-TEST.json')

    assert nn.input_neurons[4].weights == nn2.input_neurons[4].weights
    assert nn.hidden_neurons[14].weights == nn2.hidden_neurons[14].weights
    assert nn.output_neurons[4].weights == nn2.output_neurons[4].weights
    
    board = Connect4Board()

    print('nn select_move')
    nn.select_move(board)

    print('nn2 select_move')
    nn2.select_move(board)
