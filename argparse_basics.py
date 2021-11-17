import argparse

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--required", required=True,
                      help="required argument")
    parser.add_argument("--choices", choices=["choice1", "choice2"], default="choice1",
                      help="argument from list of choices")
    parser.add_argument("--int_argument", default=100, type=int,
                      help="int argument")
    parser.add_argument("--str_arg", type=str, default="default_str",
                      help="string argument")
    parser.add_argument("--float_arg", default=1.0, type=float,
                      help="float argument")
    parser.add_argument('--arg_action', action='store_true',
                      help="argument that does an action when used")
    args = parser.parse_args()

if __name__ == '__main__':
    main()