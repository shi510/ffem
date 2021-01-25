import argparse
import os

import generate_bbox.trillion_pairs as trillion_pairs
import generate_bbox.vggface as vggface
import generate_bbox.rfw as rfw


parser = argparse.ArgumentParser()
sub_parser = parser.add_subparsers(dest='cmd')
trillon_parser = sub_parser.add_parser('trillion_pairs')
trillon_parser.add_argument('--lndmk')
trillon_parser.add_argument('--output')
vggface_parser = sub_parser.add_parser('vggface')
vggface_parser.add_argument('--lndmk')
vggface_parser.add_argument('--output')
rfw_parser = sub_parser.add_parser('rfw')
rfw_parser.add_argument('--lndmk')
rfw_parser.add_argument('--output')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.cmd == 'trillion_pairs':
        if os.path.isfile(args.lndmk):
            trillion_pairs.save_bbox_to_json(args.lndmk, args.output)
        else:
            print('{} not exists.'.format(args.lndmk))
    elif args.cmd == 'vggface':
        if os.path.isfile(args.lndmk):
            vggface.save_bbox_to_json(args.lndmk, args.output)
        else:
            print('{} not exists.'.format(args.lndmk))
    elif args.cmd == 'rfw':
        if os.path.isfile(args.lndmk):
            rfw.save_bbox_to_json(args.lndmk, args.output)
        else:
            print('{} not exists.'.format(args.lndmk))
