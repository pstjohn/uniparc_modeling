import os
import json
import argparse

parser = argparse.ArgumentParser(description='BERT model training')
parser.add_argument('n', type=int, help='worker id')
parser.add_argument('--hostlist', default='', help='file containing list of hosts')
parser.add_argument('--port', '-p', default=22834, help='communication port')
parser.add_argument('--modelName', default='bert', help='model name for directory saving')

arguments = parser.parse_args()

with open(arguments.hostlist) as file:
    hosts = file.readlines()

num_hosts = len(hosts)
n = arguments.n
    
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": [f"{worker.strip()}:{arguments.port}" for worker in hosts],
    },
   "task": {"type": "worker", "index": n}
})

print(os.environ["TF_CONFIG"], flush=True)
