## Evaluating concurrent reinforcement learning algorithms


#### Environment Setup
*Note: These instructions work on Linux but may need slight modification to work on Windows / Mac*

Create python virtual env (currently using Python 3.6 but other versions may work):
```bash
python3 -m venv
```

Activate virtual env:
Create python virtual env (currently using Python 3.6 but other versions may work):
```bash
source venv/bin/activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

#### Running the code
Run as follows (modifying arguments as desired)

```bash
python3 run.py --agent=dqn --num-threads=2
```

*Note: Use --agent=dqn for Parallel DQN and --agent=a3c for Asynchronous Actor Critic algorithm*