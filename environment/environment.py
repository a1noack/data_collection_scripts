import pygame, random, time, numpy as np
from multiprocessing import Pipe
import os

class PacmanEnv():
    def __init__(self, num_episodes=4, scale=5, move_left=.8, is_beneficial=.8, speed=5, update_freq=20,
                 reward=200, punishment = 500, time_bw_epi=5, display=True, hangtime=2, deviate=1,
                 win_value=-1):
        #set width and height of gameboard
        self.width = int(scale*100)
        self.height = int(self.width/3)

        #initial pacman and character coordinates and set size of components
        self.character_size = int(self.width / 8)
        self.x_pacman, self.y_pacman = (int((self.width / 2) - (self.character_size / 2)), int((self.height / 2) - (self.character_size / 2)))
        self.x_entity1, self.y_entity1 = (int((1 * self.width / 5) - (self.character_size / 2)), int((self.height / 2) - (self.character_size / 2)))
        self.x_entity2, self.y_entity2 = (int((4 * self.width / 5) - (self.character_size / 2)), int((self.height / 2) - (self.character_size / 2)))
        self.distance_to_entities = abs(self.x_pacman-self.x_entity1)

        #if the random float is less than .8, then this episode's right state is beneficial
        self.b = is_beneficial
        self.l = move_left
        self.is_beneficial = random.uniform(0,1) <= self.b
        self.move_left = random.uniform(0, 1) <= self.l
        self.update_freq = update_freq #this is approximately the average update frequency from tests
        # self.clock = pygame.time.Clock()
        self.speed_initial = speed
        self.pixels_per_second = self.update_freq * self.speed_initial
        #I'm not sure why I have to multiply this value by 2. idk
        self.episode_duration = (self.distance_to_entities/self.pixels_per_second) + hangtime
        self.hangtime = hangtime
        self.epi_type = 0b00
        self.num_episodes = num_episodes

        self.win_value = move_left*is_beneficial*reward + (1-move_left)*is_beneficial*(-punishment) + move_left*(1-is_beneficial)*(-punishment) + (1-move_left)*(1-is_beneficial)*(reward)
        self.win_value = self.win_value*num_episodes

        if (win_value != -1):
            self.win_value = win_value
        # print("episode duration",self.episode_duration)
        if(self.move_left):
            self.speed = -self.speed_initial
        else:
            self.speed = self.speed_initial
        self.reward, self.punishment = (reward, punishment) # .75(+200) + .25(-400) = +50. A naive approach favors going toward state of interest in absence of info about state value.
        if(display):
            self.score = 0
            self.time_between_episodes = time_bw_epi #seconds
        self.last_change = 0
        self.num_wins = int(((self.win_value/self.num_episodes + self.punishment)*self.num_episodes)/(self.reward+self.punishment)) + deviate
        self.num_losses = self.num_episodes - self.num_wins
        self.sequence = [1]*self.num_wins
        self.sequence = np.asarray(self.sequence + [0]*self.num_losses)
        np.random.shuffle(self.sequence)
        self.cur_dir = os.getcwd()

    #returns estimated duration of episode in seconds
    def get_duration(self):
        return self.episode_duration

    #initialize other stuff
    def load_stuff(self):
        # initialize gameboard
        self.game_display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        # pacman yellow rgb values
        self.pacman_yellow = (255, 238, 0)

        # load images
        self.score_font = pygame.font.SysFont("arial", int(self.width * (2 / 30)), bold=True)
        self.secondary_score_font = pygame.font.SysFont("arial", int(self.width * (1.5 / 30)), bold=False)
        self.pacman_image = pygame.transform.scale(pygame.image.load(self.cur_dir+'/environment/images/pacman.png'), (self.character_size, self.character_size))
        if (self.move_left):
            self.pacman_image = pygame.transform.flip(self.pacman_image, 1, 0)
        self.maze = pygame.transform.scale(
            pygame.image.load(self.cur_dir+'/environment/images/pacman_maze.png'),
            (self.width, self.height))
        self.ghost_image = pygame.transform.scale(
            pygame.image.load(self.cur_dir+'/environment/images/pacman_pinky.png'),
            (self.character_size, self.character_size))
        self.strawberry_image = pygame.transform.scale(pygame.image.load(
            self.cur_dir+'/environment/images/pacman_strawberry.png'),
                                                       (self.character_size, self.character_size))
        self.score_text = self.score_font.render("{0}".format(self.score), 1, self.pacman_yellow)

        self.vert = pygame.transform.scale(
            pygame.image.load(self.cur_dir+'/environment/images/circle.png'),
            (int(self.character_size * 1.2), int(self.character_size * 1.2)))
        self.vert = pygame.transform.rotate(self.vert, 90)

    # reposition pacman
    def update_pacman(self):
        self.game_display.blit(self.pacman_image, (self.x_pacman,self.y_pacman))

    #update entity positions and background
    def update_constants(self):
        self.game_display.blit(self.maze, (0, 0))
        if(self.is_beneficial):
            self.game_display.blit(self.strawberry_image, (self.x_entity1, self.y_entity1))
            self.game_display.blit(self.ghost_image, (self.x_entity2, self.y_entity2))
        else:
            self.game_display.blit(self.strawberry_image, (self.x_entity2, self.y_entity2))
            self.game_display.blit(self.ghost_image, (self.x_entity1, self.y_entity1))

    #update the type of this episode
    def record_episode_type(self):
        if(self.is_beneficial and self.move_left):
            self.epi_type = 0
        elif(not self.is_beneficial and self.move_left):
            self.epi_type = 1
        elif(self.is_beneficial and not self.move_left):
            self.epi_type = 2
        else:
            self.epi_type = 3

    #update score text graphic
    def update_remaining_epi(self, multi_ep, epi_rem):
        self.game_display.blit(self.maze, (0, 0))

        str = "Score: {0}".format(self.score)
        score_to_beat = "High score: {0}".format(int(self.win_value))
        str2 = "{0} episodes remaining".format(epi_rem)

        self.episodes_remaining_text = self.secondary_score_font.render(str2, 1, self.pacman_yellow)
        self.score_text = self.score_font.render(str, 1, self.pacman_yellow)
        self.win_value_text = self.secondary_score_font.render(score_to_beat, 1, self.pacman_yellow)

        self.game_display.blit(self.score_text, (30, 5))
        self.game_display.blit(self.episodes_remaining_text, (30, 70))
        self.game_display.blit(self.win_value_text, (350, 10))

    #returns true if pacman is near either of the two entities
    def overlapping(self):
        if(abs(self.x_pacman-self.x_entity1) < .5*self.character_size or abs(self.x_pacman-self.x_entity2) < .5*self.character_size):
            return True
        else:
            return False

    def nearby(self):
        if (abs(self.x_pacman - self.x_entity1) < 1.5 * self.character_size or abs(
                    self.x_pacman - self.x_entity2) < 1.5 * self.character_size):
            return True
        else:
            return False

    #resets the environment for a new episode
    def reset_env(self):
        # if is_beneficial, the reward state is on the left. If not, the reward state is on the right.
        # if move_left, pacman moves left in this episode. If not, pacman moves right.
        # therefore, two of the four game states incur reward. is_ben && move_left, and !is_ben && !move_left
        if ((self.is_beneficial and self.move_left) or (not self.is_beneficial and not self.move_left)):
            # self.score += self.reward
            self.last_change = self.reward
        else:
            # self.score -= self.punishment
            self.last_change = -self.punishment

        # roll dice to determine new gamestate. (four possibilities)
        # self.is_beneficial = random.uniform(0, 1) <= self.b
        # self.move_left = random.uniform(0, 1) <= self.l

    #recenters pacman on screen
    def reposition_pacman(self):
        self.x_pacman = int((self.width / 2) - (self.character_size / 2))

    def set_speed_direction(self):
        # reset pacman
        self.pacman_image = pygame.transform.scale(pygame.image.load(self.cur_dir+'/environment/images/pacman.png'),(self.character_size, self.character_size))
        if (self.move_left):
            self.pacman_image = pygame.transform.flip(self.pacman_image, 1, 0)
        self.reposition_pacman()
        if (self.move_left):
            self.speed = -self.speed_initial
        else:
            self.speed = self.speed_initial

    #checks if the player has won the game
    def did_win(self):
        if(self.score >= self.win_value):
            return True
        return False

    #simluates num_episodes of the simulation
    def simulate(self, num_episodes=10):
        clock = pygame.time.Clock()
        pygame.init()
        self.load_stuff()
        #the game is not initially crashed
        self.num_episodes=num_episodes
        crashed = False
        counter = 0
        record_times = []
        while not crashed and self.num_episodes>0:
            if (counter == 50):
                l = np.average(np.diff(record_times))
                print("average receive frequency:", 1 / l, "Hz")
                record_times = []
                counter = 0
            current_time = time.time()
            record_times.append(current_time)
            counter += 1
            #check if game has crashed and exit loop if it has
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    crashed = True
                    print("Game killed.")
            self.update_constants()

            #update pacman's position according to speed variable
            self.x_pacman += self.speed
            self.update_pacman()

            #update game state for new episode
            if(self.overlapping()):
                self.reset_env()
                self.num_episodes-=1
                time.sleep(self.time_between_episodes)
            self.update_remaining_epi(multi_ep=True)
            pygame.display.update()
            #update pygame state
            clock.tick(self.update_freq)
        pygame.quit()
        quit()

    #obscures the initial direction of pacman
    def obscure_dir(self):
        self.game_display.blit(self.vert, (self.x_pacman-7,self.y_pacman-8))

    #defines screen after each episode
    def immediate_change(self):
        self.game_display.blit(self.maze, (0, 0))
        if(self.last_change > 0):
            str2 = "+ {0} points".format(self.last_change)
        else:
            str2 = "{0} points".format(self.last_change)

        self.last_change_text = self.score_font.render(str2, 1, self.pacman_yellow)

        self.game_display.blit(self.last_change_text, (self.width/2-100, self.height/2-50))

    #defines screen to be displayed at the end of the trial
    def display_win_screen(self):
        self.game_display.blit(self.maze, (0, 0))
        if(self.did_win()):
            str = "YOU WON!!!   WIN CODE: 3412"
        else:
            str = "You lose.."
        win_text = self.score_font.render(str, 1, self.pacman_yellow)
        self.game_display.blit(win_text, (50, 50))

    #simulate one episode and wait ttw seconds before moving pacman toward either goal
    def simulate_one_epi(self, ani_pipeend, display_freq, epi_rem, cur_epi, control, win_lose, score_queue, current_score):
        # initialize pygame here because apparently it doesn't work if done in __init__
        pygame.init()
        self.is_beneficial = random.uniform(0, 1) <= self.b
        won = False
        self.overlap = False
        crashed = False
        #the clock isn't picklable so maybe it'll work to create it here??
        clock = pygame.time.Clock()

        self.load_stuff()
        self.score = current_score
        #show the score and remaining episode screen
        end_time = time.time() + 1.75
        while time.time() < end_time:
            self.update_remaining_epi(multi_ep=True, epi_rem=epi_rem)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Game killed.")
            pygame.display.update()
        # counter = 0
        self.update_constants()
        self.update_pacman()
        #make sure the direction pacman is facing cannot be determined
        self.obscure_dir()
        #aligns pacman with preconceived episode type from the
        if (control != -1):
            # print("win/lose:", win_lose)
            #if choice to operate predictably, and
            if(win_lose == 1):
                if(self.is_beneficial == True):
                    self.move_left = True
                else:
                    self.move_left = False
            else:
                if (self.is_beneficial == True):
                    self.move_left = False
                else:
                    self.move_left = True
            self.set_speed_direction()
        # update the type of this episode
        self.record_episode_type()
        #pause here until the animation receives the start signal from the mwm
        ani_pipeend.recv()
        #as soon as the episode commences, send the recorder the correct episode type
        ani_pipeend.send([self.epi_type])
        pygame.display.update()
        print("\tCommencing animation at ",time.time())
        #need this loop for the display to display
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
                print("Game killed.")
        #wait to display the episode hangtime seconds
        time.sleep(self.hangtime)

        counter = 0
        record_times = []
        while not crashed and not self.overlap:
            if(display_freq):
                #this code below finds the frames per second that this simulation is updating at
                if (counter == 10):
                    l = np.average(np.diff(record_times))
                    print("\t\tanimation:", 1 / l, "Hz")
                    record_times = []
                    counter = 0
                current_time = time.time()
                record_times.append(current_time)
                counter += 1
            # check if game has crashed and exit loop if it has
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    crashed = True
                    print("Game killed.")
            self.update_constants()
            # update pacman's position according to speed variable
            self.x_pacman += self.speed
            self.update_pacman()
            # update game state for new episode
            if (self.overlapping()):
                print("\tClosing animation at ", time.time())
                self.reset_env()
                self.overlap = True
                # self.num_episodes -= 1
                self.immediate_change()
                self.score = current_score + self.last_change
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Game killed.")
                time.sleep(.5)
            # self.update_remaining_epi(multi_ep=True, epi_rem=epi_rem)
            clock.tick(self.update_freq)
            pygame.display.update()
        if (epi_rem == 1):
            self.display_win_screen()
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Game killed.")
            time.sleep(5)
        pygame.quit()
        score_queue.put([self.score])

    # simulate one episode and wait ttw seconds before moving pacman toward either goal
    def simulate_multi_epi(self, ani_pipeend, display_freq, control): #send sequence of wins and losses,
        # initialize pygame here because apparently it doesn't work if done in __init__
        epi_rem = self.num_episodes
        pygame.init()
        cur_epi = 0
        clock = pygame.time.Clock()
        self.load_stuff()
        crashed = False
        self.score = 0
        # pause here until the animation receives the start signal from the mwm and the reader passes the startup spike
        ani_pipeend.recv()
        while cur_epi < self.num_episodes:
            first_pass = True
            win_lose = self.sequence[cur_epi]
            print("Episode",cur_epi,"/",self.num_episodes-1,"  ",win_lose)
            self.is_beneficial = random.uniform(0, 1) <= self.b
            won = False
            self.overlap = False
            # show the score and remaining episode screen
            end_time = time.time() + 1.85
            while time.time() < end_time:
                self.update_remaining_epi(multi_ep=True, epi_rem=self.num_episodes-cur_epi)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Game killed.")
                pygame.display.update()
            # counter = 0
            self.update_constants()
            self.update_pacman()
            # make sure the direction pacman is facing cannot be determined
            self.obscure_dir()
            # aligns pacman with preconceived episode type from the
            if (control != -1):
                # print("win/lose:", win_lose)
                # if choice to operate predictably, and
                if (win_lose == 1):
                    if (self.is_beneficial == True):
                        self.move_left = True
                    else:
                        self.move_left = False
                else:
                    if (self.is_beneficial == True):
                        self.move_left = False
                    else:
                        self.move_left = True
                self.set_speed_direction()
            # update the type of this episode
            self.record_episode_type()
            pygame.display.update()
            # as soon as the episode commences, send the recorder the correct episode type
            ani_pipeend.send([self.epi_type])
            ani_pipeend.recv()
            print("\tCommencing animation at ", time.time())
            # need this loop for the display to display
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    crashed = True
                    print("Game killed.")
            # wait to display the episode hangtime seconds
            time.sleep(self.hangtime)
            action_time = time.time()

            counter = 0
            record_times = []
            while not crashed and not self.overlap:
                if (display_freq):
                    # this code below finds the frames per second that this simulation is updating at
                    if (counter == 10):
                        l = np.average(np.diff(record_times))
                        print("\t\tanimation:", int(1 / l), "Hz")
                        record_times = []
                        counter = 0
                    current_time = time.time()
                    record_times.append(current_time)
                    counter += 1
                # check if game has crashed and exit loop if it has
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        crashed = True
                        print("Game killed.")
                self.update_constants()
                # update pacman's position according to speed variable
                self.x_pacman += self.speed
                self.update_pacman()


                #receive prediction from mwm_pipend
                if(self.nearby() & first_pass):
                    prediction = ani_pipeend.recv()[0]
                    #if prediction = 0, that means the CNN predicted that Pacman is moving in the wrong direction ie 'loss'
                    if(prediction == 0):
                        self.move_left = not self.move_left
                        self.set_speed_direction()
                    #only check this on Pacman's first run. Once he turns right, eg, he will not be able to turn left again.
                    first_pass = False


                # update game state for new episode
                if (self.overlapping()):
                    print("\tClosing animation at ", time.time())
                    cur_epi += 1
                    self.reset_env()
                    self.reposition_pacman()
                    self.overlap = True
                    # self.num_episodes -= 1
                    self.immediate_change()
                    # self.score = current_score + self.last_change
                    self.score += self.last_change
                    pygame.display.update()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("Game killed.")
                    time.sleep(.25)
                    ani_pipeend.send([action_time])
                    self.game_display.blit(self.maze, (0, 0))
                    time.sleep(.25)
                # self.update_remaining_epi(multi_ep=True, epi_rem=epi_rem)
                clock.tick(self.update_freq)
                pygame.display.update()
            if (cur_epi == self.num_episodes):
                self.display_win_screen()
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Game killed.")
                time.sleep(5)
        pygame.quit()


