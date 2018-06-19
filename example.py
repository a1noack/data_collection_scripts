import multiprocessing, pygame, time

epi_dur = 4
num_episodes = 4

def run(start_time):
    pygame.init()
    crashed = False
    game_display = pygame.display.set_mode((200, 200))
    end_time = start_time + epi_dur
    while not crashed and time.time() < end_time:
        for event in pygame.event.get():
            if event == pygame.QUIT:
                crashed = True
        pygame.draw.circle(game_display, (255,255,0), (50, 50), 30, 30)
        pygame.display.update()
    pygame.quit()

def wait(wait_time, iteration):
    print("iteration:",iteration,"/",num_episodes)
    time.sleep(wait_time)
    print("\tdone...")

for i in range(1,num_episodes+1):
    start_time = time.time()
    process1 = multiprocessing.Process(target=run, args=(start_time,))
    process2 = multiprocessing.Process(target=wait, args=(epi_dur,i,))

    process1.start()
    process2.start()

    process1.join()
    process2.join()