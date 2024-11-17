import argparse
import os
import json
import shutil
from glob import glob

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dev', action='store_true')
    args = args.parse_args()
    prompts = list(json.load(open('data/wide_image_prompts.json')).keys())
    if args.dev:
        prompts = prompts[0]

    for i, prompt in enumerate(prompts):
        print(f'Running prompt: {prompt}')
        cmd = f'python main.py --app wide_image --prompt {prompt}'

        if args.dev:
            cmd += ' --tag wide_image --save_dir_now'
        else:
            cmd += f' --tag wide_image_{i}'

        os.system(cmd)

        if not args.dev:
            os.makedirs(f'results/wide_image', exist_ok=True)

            for f in glob(f'outputs/wide_image_{i}/*.png'):
                shutil.copyfile(f, f'results/wide_image/{f.split("/")[-1]}')

# python run_task1.py && python eval.py --fdir1 ./results/wide_image/ --app wide_images