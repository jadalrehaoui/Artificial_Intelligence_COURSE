# Artificial Intelligence Course
*Jad Alrehaoui* 

## Purspose

Train an agent to find the best path for a treasure. The game is called Treasure hunt and the purpose is to find the treasure before the pirate find it. I designed and intelligent agent for a non player character to represent a pirate.
The pirate agent's goal is to find the treasure before the human player. this is commonly called a pathfinding problem, as the agent I create will need to find a path towards its goal.

## Code

This project was provided to the students of this class entirely except for the q
**qtrain** function. 
Here's is the code I worked on: 
```
def qtrain(model, maze, **opt):
    # exploration factor
    global epsilon 
    # number of epochs
    n_epoch = opt.get('n_epoch', 15000)
    # maximum memory to store episodes
    max_memory = opt.get('max_memory', 1000)
    # maximum data size for training
    data_size = opt.get('data_size', 50)
    # start time
    start_time = datetime.datetime.now()
    # Construct environment/game from numpy array: maze (see above)
    qmaze = TreasureMaze(maze)
    # Initialize experience replay object
    experience = GameExperience(model, max_memory=max_memory)
    win_history = []   # history of win/lose game
    hsize = qmaze.maze.size//2   # history window size
    win_rate = 0.0
    
    # pseudocode:
    # For each epoch:
    for epoch in range(n_epoch):
        #print("Start")
    #    Agent_cell = randomly select a free cell
        agent_cell = np.random.randint(0, high = 7, size = 2)
    #    Reset the maze with agent set to above position
        #print("reset")
        qmaze.reset([0,0])
    #    Hint: Review the reset method in the TreasureMaze.py class.
    #    envstate = Environment.current_state
        #print("observe")
        envstate = qmaze.observe()
        loss = 0
        n_episodes = 0
    #    Hint: Review the observe method in the TreasureMaze.py class.
    #    While state is not game over:
        while qmaze.game_status() == "not_over":
    #        previous_envstate = envstate
            previous_envstate = envstate
    #        Action = randomly choose action (left, right, up, down) either by exploration or by exploitation
            #print("actions")
            valid_actions = qmaze.valid_actions()
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(envstate))
    #        envstate, reward, game_status = qmaze.act(action)
            #print("act")
            envstate, reward, game_status = qmaze.act(action)
            n_episodes += 1
    #    Hint: Review the act method in the TreasureMaze.py class.
    #        episode = [previous_envstate, action, reward, envstate, game_status]
            episode = [previous_envstate, action, reward, envstate, game_status]
            #print("remember 2")
            experience.remember(episode)
    #        Store episode in Experience replay object
    #    Hint: Review the remember method in the GameExperience.py class.
    #        Train neural network model and evaluate loss
    #    Hint: Call GameExperience.get_data to retrieve training data (input and target) and pass to model.fit method 
    #          to train the model. You can call model.evaluate to determine loss.
            #print("inputs get data")
            inputs, targets = experience.get_data()
            #print("fit")
            history = model.fit(inputs, targets, epochs=8, batch_size=24, verbose=0)
            #print("eval")
            #loss = model.evaluate(inputs, targets)
    #    If the win rate is above the threshold and your model passes the completion check, that would be your epoch.
            if episode[4] == 'win':
                win_history.append(1)
                win_rate = sum(win_history)/len(win_history)
                break
            elif episode[4] == 'lose':
                win_history.append(0)
                win_rate = sum(win_history)/len(win_history)
                break
            
        if win_rate > epsilon:
            #print('win rate is greater than epsilon.')
            if completion_check(model, qmaze) == True:
                print('completion_check passes')

    #Print the epoch, loss, episodes, win count, and win rate for each epoch
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
        # We simply check if training has exhausted all free cells and if in all
        # cases the agent won.
        if win_rate > 0.9 : epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break
    
    
    # Determine the total time for training
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)

    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds
```

## What do computer scientists do and why does it matter? 
As a computer scientist myself, I think we play a huge role in everyone's with a phone day to day life. We are responsible of any online service and any offline system in charge of data and keeping things organized. 
I think the best trait a computer scientist has is being self organized. 
We need to keep ourselves up to date because technology doesn't stop growing and so do we. We are always learning new things and new ways of doing things. 
The best thing a computer scientist learns is how to be efficient, because when we write code or design software we need to use it to 100% so it needs to be as efficient as it could.

## How do I approach problems ?
Problems and failures are part of success, they are not breakpoints or something that makes me give up. If I have a problem that means I need to fix it and search for a solution and work my way to find it. ***I always deal with problems when coding, either someone else's code or mine.*** The best way I found solving a problem is refering to the documentation of the code or the requirement and seeing how the problem started. Sometimes it might be a design problem where the solution might be costly, so I need to be as flexible as I can to work around. Sometimes it cannot be fixed but it can be worked around. 

## What are my ethical responsibilities to the end user and the organization ?
My ethical responsibilities to the end user are respecting their privacy and not using my privilege to access information for some personal reasons. 

My ethical responsibilities to the organization are being as transparent as I can when working on tasks, don't abuse the privilege of access organization information.

*Thanks for reading this*
