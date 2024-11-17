import argparse
import os
import json
import shutil
from glob import glob

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dev', action='store_true')
    args = args.parse_args()
    prompts = json.load(open('data/ambiguous_image_prompts.json'))
    for i in prompts.keys():
        prompts[i] = (prompts[i]['instance_prompt'], prompts[i]['canonical_prompt'])
    
    prompts = list(prompts.values())

    if args.dev:
        prompts = [prompts[0]]
    
    for i in range(len(prompts)):
        prompt = prompts[i]
        print(f'Running prompt: {prompt}')
        cmd = f'python main.py --app ambiguous_image'
        cmd += f' --prompts "{prompt[0]}" "{prompt[1]}"'

        if args.dev:
            cmd += ' --tag ambiguous_image --save_dir_now'
        else:
            cmd += f' --tag ambiguous_image_{i}'

        os.system(cmd)

        if not args.dev:
            os.makedirs(f'results/ambiguous_image', exist_ok=True)

            for f in glob(f'outputs/ambiguous_image_{i}/results/*.png'):
                name = f.split("/")[-1]
                if 'final' in name or 'view' in name:
                    continue
                shutil.copyfile(f, f'results/ambiguous_image/{name}')

# python run_task2.py && python eval.py --fdir1 ./results/ambiguous_image/ --app ambiguous_images