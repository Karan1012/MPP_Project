## Evaluating concurrent reinforcement learning algorithms


#### Environment Setup
*Note: These instructions work on Linux but may need slight modification to work on Windows / Mac*

Create python virtual env (currently using Python 3.6 but other versions may work):
```bash
python3 -m venv venv
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

*Note: Use --agent=dqn for Parallel DQN, --agent=a3c for Asynchronous Actor Critic, and --agent=dynaq to run Dyna-Q algorithm*


#### Expected output

You will see the current episode number, average score, and elapsed runtime as output as the model trains, similar to the output below. 
Once the model reaches an average score of 200, the execution will terminate. 

```bash
Thread: 0, Episode 1	Average Score: -333.14, Runtime: 00:00:00
Thread: 0, Episode 2	Average Score: -268.88, Runtime: 00:00:00
Thread: 0, Episode 3	Average Score: -340.82, Runtime: 00:00:01
Thread: 0, Episode 4	Average Score: -279.05, Runtime: 00:00:02
Thread: 0, Episode 5	Average Score: -297.42, Runtime: 00:00:03
...
```


For Dyna-Q, the world loss is also output as the world model trains, to show that the world model is in fact being trained over time (loss should decrease over time).