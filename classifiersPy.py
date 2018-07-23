#!/usr/bin/env python3
# encoding: utf-8

import sys
import argparse
import logging
from neural_network.neural_network import NeuralNet


module = sys.modules['__main__'].__file__
log = logging.getLogger(module)


def parse_command_line(argv):
    """Parse command line argument. See -h option
    :param argv: arguments on the command line must include caller file name.
    """
    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=module,
                                     formatter_class=formatter_class)
    parser.add_argument("-v", "--verbose", dest="verbose_count",
                        action="count", default=0,
                        help="Aumenta la verbosidad del registro para cada ocurrencia.")
    parser.add_argument("-nN", "--neuralNet",
                        help="Usar red neuronal.")
    parser.add_argument("-bN", "--bayesNet",
                        help="Usar red bayesiana.")
    parser.add_argument('-o', metavar="output",
                        type=argparse.FileType('r'), default=sys.stdout,
                        help="Redirigir la salida a un archivo")
    parser.add_argument('-f', metavar="file",
                        type=argparse.FileType('w'), default=sys.stdout,
                        help="Archivo con datos de entrada (csv)")
    parser.add_argument('-e', metavar="epoch",
                        type=int, default=0,
                        help="Epoch for Neural Network")
    parser.add_argument('-nI', metavar="n_input",
                        type=int, default=0,
                        help="n_input")
    parser.add_argument('-nH', metavar="n_hidden",
                        type=int, default=0,
                        help="n_hidden")
    parser.add_argument('-nO', metavar="n_output",
                        type=int, default=0,
                        help="n_output")
    parser.add_argument('input', metavar="input", nargs='+', help="Cualquier entrada...")
    arguments = parser.parse_args(argv[0:])

    log.setLevel(max(3 - arguments.verbose_count, 0) * 10)
    return arguments


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG,
                        format='%(name)s (%(levelname)s): %(message)s')
    try:
        arguments = parse_command_line(sys.argv)
        NeuralNet()
        print(arguments.f)
        # Do something with arguments.
    except KeyboardInterrupt:
        log.error('Programa interrumpido!')
    finally:
        logging.shutdown()

if __name__ == "__main__":
    sys.exit(main())