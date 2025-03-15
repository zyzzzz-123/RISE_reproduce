import os
import json
import numpy as np
import argparse
import ipdb


parser = argparse.ArgumentParser(description='RISE evaluation analysis')
parser.add_argument('--input_file', type=str, default="eval.json", help="input file")
parser.add_argument('--output_dir', type=str, default="eval_ana", help="output directory")  
parser.add_argument('--turn', type=int, default=1, help="turn number")  
args = parser.parse_args()  

print(args)


def eval_ana_five_turn(input_file, output_dir):
    with open(input_file, 'r') as f:
        data = json.load(f)
    # print(data)
    all_rewards = []
    
    for index, result in data.items():
        rewards = result['turn_history']['rewards']
        reward_list = []
        
        for i in range(len(rewards)):
            reward_list.append(rewards[str(i)])
        
        all_rewards.append(reward_list)
        
    # turn all_rewards into numpy array
    all_rewards = np.array(all_rewards)
    all_rewards = all_rewards.squeeze()

    m1_t1 = all_rewards[:, 0]
    m1_success = np.sum(m1_t1 == 1) / len(m1_t1)
    
    m5_t1_success_rate = np.sum(np.any(all_rewards == 1, axis=1)) / len(all_rewards)
    
    print(f"m1@t1: {m1_success}")
    print(f"m5@t1: {m5_t1_success_rate}")
    
    # write the analysis to a file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, 'eval_ana.txt'), 'w') as f:
        f.write(f"m1@t1: {m1_success}\n")
        f.write(f"m5@t1: {m5_t1_success_rate}\n")
    

def eval_ana_one_turn(input_file, output_dir):
    with open(input_file, 'r') as f:
        data = json.load(f)
    # print(data)
    all_rewards = []
    
    for index, result in data.items():
        rewards = result['turn_history']['rewards']
        reward_list = []
        
        for i in range(len(rewards)):
            reward_list.append(rewards[str(i)])
        
        all_rewards.append(reward_list)
        
    # ipdb.set_trace()
    # turn all_rewards into numpy array
    all_rewards = np.array(all_rewards)
    all_rewards = all_rewards.squeeze()

    success_in_row = np.any(all_rewards == 1, axis=1)
    
    m1_t5 = np.mean(success_in_row)
    
    print(f"m1@t5: {m1_t5}")
    
    # write the analysis to a file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, 'eval_ana.txt'), 'w') as f:
        f.write(f"m1@t5: {m1_t5}")

    
    
if __name__ == "__main__":
    if args.turn == 5:
        eval_ana_five_turn(args.input_file, args.output_dir)
    elif args.turn == 1:
        eval_ana_one_turn(args.input_file, args.output_dir)
