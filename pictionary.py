import pygame
import os
import numpy as np
import argparse
import time
import cv2 as cv
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

pygame.init()
pygame.mouse.set_visible(False)

class Game:
    def __init__(self):
        # setting for the game itself
        # list of players to iterate over sorted to iterate over in a "game manner"
        # thus: (team_1_player_1, team_2_player_1, team_1_player2, team_2_plyaer_2)
        # such that playing team changes each turn 
        self.players = [] 
        self.teams = {"team 1": [ ], "team 2": [ ]} # dict of teams with list of players in each team
        self.teamcount = 2
        self.playercount = 4 
        self.current_score = {"team 1": 0, "team 2": 0}
        self.rounds_per_team = 2
        self.seconds_per_drawing = 40

        self.items = []
        item_file = open("draw_items.txt", "r")
        for line in item_file:
            line = line.replace("\n", "")
            self.items.append(line)
        item_file.close()

        

        # load images needed for the GUI
        self.mouse_image = pygame.image.load(os.path.join("assets", "pencil.png"))
        self.mouse_scale_width = 40
        self.mouse_scale_height = 40
        self.mouse = pygame.transform.scale(self.mouse_image, (self.mouse_scale_width,self.mouse_scale_height))


        # menu GUI
        self.logo_image = pygame.image.load(os.path.join("assets", "logo.png"))
        self.logo = pygame.transform.scale(self.logo_image, (500,250))

        self.start_game_button_image = pygame.image.load(os.path.join("assets", "start_game_button.png"))
        self.start_game_button_width = 200
        self.start_game_button_height = 100
        self.start_game_button = pygame.transform.scale(self.start_game_button_image, (self.start_game_button_width,self.start_game_button_height))

        # create game GUI
        self.font = pygame.font.Font(None, 32)


        # Settings for the game engine
        self.display_width = 900
        self.display_height = 500
        self.display = pygame.display.set_mode((self.display_width,self.display_height))
        self.display_title = pygame.display.set_caption("Pictionary!")
        self.background_color = (255, 255, 255) # white
        self.draw_color = (0, 0, 0) # black
        self.fps = 60
        self.current_drawing = []

        # settings for the handtracking 

        # margin of the screen side we do not use
        self.margin = 0.2 

        # use half of the margin for each side
        self.draw_width_offset = 0.5 * self.margin
        self.draw_height_offset = 0.5 * self.margin

        # The draw window is a subset of the frame 
        # such that we can also reach the edges of the drawing board
        self.draw_window_width = (1 - self.margin)
        self.draw_window_height =  (1 - self.margin)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # Load the gesture recognizer model
        self.model = load_model('mp_hand_gesture')

        # Load class names
        f = open('gesture.names', 'r')
        self.classNames = f.read().split('\n')
        f.close()

        
    def draw_display(self, mouse_pos):
        self.display.fill(self.background_color)

        #for pos, size in self.current_drawing:
            #pygame.draw.circle(self.display, self.draw_color, pos, radius=size)
        

        self.display.blit(self.mouse, (mouse_pos[0], mouse_pos[1] - self.mouse_scale_height))

        pygame.display.update()

    def draw(self):
        self.current_drawing = [ ]
        guessed = False
        forced_break =False

        #set the mouse to visible to enable to click on the guessed button
        pygame.mouse.set_visible(True)

        # enable camera
        cap = cv.VideoCapture(0)

        # the loading can take a while, so we set the timer when the recording starts
        timer = pygame.time.set_timer(pygame.USEREVENT, 1000)
        countdown_text, countdown = str(self.seconds_per_drawing).rjust(3), self.seconds_per_drawing
        clock = pygame.time.Clock()

        blue_hud = pygame.Rect(0, 0, self.display_width, 50)
        guessed_button = pygame.Rect(350, 5, 200, 40)

        with self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            while cap.isOpened() and guessed == False and forced_break == False:
                success, image = cap.read()
                frame_width, frame_height, c = image.shape
                className = ""
                draw_x, draw_y = 0,0
                clock.tick(self.fps)

                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    landmarks = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Firstly, retrieve the coordinates of the fingers on the frame (relative coordinates in range [0,1])
                        # the coordinates are calculated with the upper left corner as origin (0,0)

                        # in this case it should start at 0
                        if hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x > 1:
                            finger_coor_x = 0
                        elif hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x < 0:
                            finger_coor_x = 1
                        # get the finger coorindates, adjust x value because pyautogui uses a different method
                        #else: finger_coor_x = int((1 - hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x) * max_screen_width)
                        else: finger_coor_x = 1 - hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x

                        if hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y > 1:
                        #finger_coor_y = max_screen_height
                            finger_coor_y = 1
                        elif hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y < 0:
                            finger_coor_y = 0
                        #else: finger_coor_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * max_screen_height)
                        else: finger_coor_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y 


                        if finger_coor_x < self.draw_width_offset: 
                            relative_x = 0
                        elif finger_coor_x > self.draw_window_width + self.draw_width_offset:
                            relative_x = 1
                        else:
                            relative_x = (finger_coor_x - self.draw_width_offset) / self.draw_window_width


                        if finger_coor_y < self.draw_height_offset: 
                            relative_y = 0
                        elif finger_coor_y > self.draw_window_height + self.draw_height_offset:
                            relative_y = 1
                        else:
                            relative_y = (finger_coor_y - self.draw_height_offset) / self.draw_window_height

                        draw_x, draw_y = int(relative_x * self.display_width), int(relative_y * (self.display_height))
                        # this is necessary for the hud with the "guess" button
                        if draw_y < 50: draw_y = 50
                        
                        for lm in hand_landmarks.landmark:
                            lmx = int(lm.x * frame_width)
                            lmy = int(lm.y * frame_height)

                            landmarks.append([lmx, lmy])
                        # Predict gesture
                        if len(landmarks) == 21:
                            prediction = self.model.predict([landmarks])
                            # print(prediction)
                            classID = np.argmax(prediction)
                            className = self.classNames[classID]
                

                        if className == "fist" or className == "rock":
                            self.current_drawing.append(((draw_x, draw_y), 10))
                    

                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

                # Flip the image horizontally for a selfie-view display.
                image = cv.flip(image, 1)
                cv.putText(image, className, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 
                            1, (0,0,255), 2, cv.LINE_AA)

                cv.imshow('MediaPipe Hands', image)
                #if cv.waitKey(1) == ord('q'):
                    #break

                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        # if the picture was guessed then stop
                        if guessed_button.collidepoint(event.pos):
                            guessed = True
                    if event.type == pygame.USEREVENT:
                        countdown -= 1
                        if countdown > 0: 
                            countdown_text = str(countdown).rjust(3)
                        # if the time is up then also stop the loop
                        else: 
                            forced_break = True

                # display the draw screen
                #self.draw_display((draw_x, draw_y))
                self.display.fill(self.background_color)

                # draw hud and guessed button
                pygame.draw.rect(self.display, (51,153,255), blue_hud)
                pygame.draw.rect(self.display, (0,204,0), guessed_button)
                self.display.blit(self.font.render("GUESSED!", True, (255, 255, 255)), (390,15))
                self.display.blit(self.font.render(countdown_text, True, (255, 255, 255)), (600,15))

                # draw the current drawingq
                for pos, size in self.current_drawing:
                    pygame.draw.circle(self.display, self.draw_color, pos, radius=size)
                

                self.display.blit(self.mouse, (draw_x, draw_y - self.mouse_scale_height))

                pygame.display.update()
        cap.release()
        cv.destroyAllWindows()
        self.current_drawing = [ ]
        pygame.mouse.set_visible(False)

        score = 0
        if guessed:
            score = 10 + countdown
        return score




    # main game loop in which the game runs
    def run(self):
        self.display_menu()
        """
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(self.fps)
            self.display_menu()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = pygame.mouse.get_pos()
                    #self.current_drawing.append((pos, 20))
            self.draw_display(pygame.mouse.get_pos())
        pyg
        ame.quit()
        """
    
    # function to display the main menu
    # main options in menu: settings, start game
    def display_menu(self):
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(self.fps)
            pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # if the start button is pressed then call start game
                    if pos[0] >= 350 and pos[0] <= 350 + self.start_game_button_width and pos[1] >= 200 and pos[1] <= 200 + self.start_game_button_height:
                        self.start_game()            

            self.display.fill(self.background_color)
            
            self.display.blit(self.logo, (200, 0))
            self.display.blit(self.start_game_button, (350, 200))

            # always draw the mouse last 
            self.display.blit(self.mouse, (pos[0], pos[1] - self.mouse_scale_height))
            pygame.display.update()        
        
        # only use this in main loop
        pygame.quit()
    
    # adjust current settings, not a must in the game
    def adjust_settings(self):
        return 0
    
    # start a game 
    # firstly use the "create teams" function to assign team names
    # then run a loop of supply item -> 
    # draw -> guess -> reward -> again
    def start_game(self):
        self.create_teams()
        current_team = "team 1"

        continue_button = pygame.Rect(350, 400, 200, 50)

        # play the amount of rounds given
        for i in range(self.rounds_per_team):
            # iterate over the players in good order
            for player in self.players:
                # assign an item to draw
                item =  np.random.choice(self.items)

                # define a counter and display the message to the player for 5 seconds
                timer = pygame.time.set_timer(pygame.USEREVENT, 1000)
                countdown_text, countdown = "5".rjust(3), 5
                temp = True
                clock = pygame.time.Clock()
                while temp: 
                    clock.tick(self.fps)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                        if event.type == pygame.USEREVENT:
                            countdown -= 1
                            if countdown > 0: 
                                countdown_text = str(countdown).rjust(3)
                            else: 
                                countdown_text = "GO!"
                                temp = False

                    self.display.fill(self.background_color)
                    text = f"{player} ({current_team}) has to draw {item}"
                    self.display.fill(self.background_color)
                    self.display.blit(self.font.render(text, True, (0, 0, 0)), (250,200))
                    self.display.blit(self.font.render(countdown_text, True, (0, 0, 0)), (450,300))
                    pygame.display.update()


                # start the drawing
                score = self.draw()

                # display the score after a player drew
                temp = True
                while temp: 
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                            if continue_button.collidepoint(event.pos):
                                temp = False
                    self.display.fill(self.background_color)
                    text = f"{player} scored {score} points this round"
                    self.display.blit(self.font.render(text, True, (0, 0, 0)), (250,200))
                    # draw continue button
                    pygame.draw.rect(self.display, (0,204,0), continue_button)
                    self.display.blit(self.font.render("Continue", True, (255, 255, 255)), (390,405))
                    self.display.blit(self.mouse, (pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1] - self.mouse_scale_height))

                    
                    pygame.display.update()
                self.current_score[current_team] += score

                    
                # switch the teams for score calculation
                if current_team == "team 1":
                    current_team = "team 2"
                else: current_team = "team 1"


            # display the standings after each round
            team_1_score = self.current_score["team 1"]
            team_2_score = self.current_score["team 2"]
            text = f"After round {i+1}, team 1: {team_1_score} and team 2: {team_2_score}"
            temp = True
            while temp: 
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                    if event.type == pygame.MOUSEBUTTONUP and event.button == 1:                          
                        if continue_button.collidepoint(event.pos):
                            temp = False
                self.display.fill(self.background_color)
                self.display.blit(self.font.render(text, True, (0, 0, 0)), (250,200))
                # draw next round button
                pygame.draw.rect(self.display, (0,204,0), continue_button)
                self.display.blit(self.font.render("Next Round", True, (255, 255, 255)), (390,405))
                self.display.blit(self.mouse, (pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1] - self.mouse_scale_height))

                    
                pygame.display.update()

        winner = self.define_winner()
        text = f"The winner of this game is {winner} !"
        temp = True
        while temp: 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:                          
                    if continue_button.collidepoint(event.pos):
                        temp = False
            self.display.fill(self.background_color)
            self.display.blit(self.font.render(text, True, (0, 0, 0)), (250,200))
            # draw finish button
            pygame.draw.rect(self.display, (0,204,0), continue_button)
            self.display.blit(self.font.render("Finish Game", True, (255, 255, 255)), (390,405))
            self.display.blit(self.mouse, (pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1] - self.mouse_scale_height))

                    
            pygame.display.update()
            
            

        # reset the scores
        self.current_score["team 1"] = 0
        self.current_score["team 2"] = 0
                
    
    # create the teams before starting the actual game 
    def create_teams(self):
        self.players = [] 
        self.teams = {"team 1": [ ], "team 2": [ ]}
        clock = pygame.time.Clock()
        run = True

        # create rectangle input boxes
        player_1_input = pygame.Rect(300, 120, 200, 50)
        player_2_input = pygame.Rect(300, 190, 200, 50)
        player_3_input = pygame.Rect(300, 370, 200, 50)
        player_4_input = pygame.Rect(300, 440, 200, 50)
        player_1_name = "Name"
        player_2_name = "Name"
        player_3_name = "Name"
        player_4_name = "Name"
        adjust_1 = False
        adjust_2 = False
        adjust_3 = False
        adjust_4 = False
        

        start_button = pygame.Rect(650, 255, 200, 50)
        start_button_text = "START!"

        while run:
            clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if player_1_input.collidepoint(event.pos):
                        adjust_1 = True
                        adjust_2 = False
                        adjust_3 = False
                        adjust_4 = False
                    if player_2_input.collidepoint(event.pos):
                        adjust_1 = False
                        adjust_2 = True
                        adjust_3 = False
                        adjust_4 = False
                    if player_3_input.collidepoint(event.pos):
                        adjust_1 = False
                        adjust_2 = False
                        adjust_3 = True
                        adjust_4 = False
                    if player_4_input.collidepoint(event.pos):
                        adjust_1 = False
                        adjust_2 = False
                        adjust_3 = False
                        adjust_4 = True
                    if start_button.collidepoint(event.pos):
                        self.players.append(player_1_name)
                        self.players.append(player_3_name)
                        self.players.append(player_2_name)
                        self.players.append(player_4_name)
                        self.teams["team 1"].append(player_1_name)
                        self.teams["team 1"].append(player_2_name)
                        self.teams["team 2"].append(player_3_name)
                        self.teams["team 2"].append(player_4_name)
                        run = False
                if event.type == pygame.KEYDOWN:
                    if adjust_1:
                        if event.key == pygame.K_BACKSPACE:
                            player_1_name = player_1_name[:-1]
                        else:
                            player_1_name += event.unicode
                    if adjust_2:
                        if event.key == pygame.K_BACKSPACE:
                            player_2_name = player_2_name[:-1]
                        else:
                            player_2_name += event.unicode
                    if adjust_3:
                        if event.key == pygame.K_BACKSPACE:
                            player_3_name = player_3_name[:-1]
                        else:
                            player_3_name += event.unicode
                    if adjust_4:
                        if event.key == pygame.K_BACKSPACE:
                            player_4_name = player_4_name[:-1]
                        else:
                            player_4_name += event.unicode
                              

            self.display.fill(self.background_color)
            s = pygame.display.get_surface()

            # display the fields that need to be filled in
            pygame.draw.rect(self.display, (51,153,255), (100, 50, 150, 50), 2)
            s.fill(pygame.Color((51,153,255)), (100, 50, 150, 50))
            self.display.blit(self.font.render("Team 1", True, (255, 255, 255)), (130,60))
            pygame.draw.rect(self.display, (51,153,255), (100, 120, 150, 50), 2)
            s.fill(pygame.Color((51,153,255)), (100, 120, 150, 50))
            self.display.blit(self.font.render("Player 1", True, (255, 255, 255)), (130,130))
            pygame.draw.rect(self.display, (51,153,255), (100, 190, 150, 50), 2)
            s.fill(pygame.Color((51,153,255)), (100, 190, 150, 50))
            self.display.blit(self.font.render("Player 2", True, (255, 255, 255)), (130,200))

            pygame.draw.rect(self.display, (51,153,255), (100, 300, 150, 50), 2)
            s.fill(pygame.Color((51,153,255)), (100, 300, 150, 50))
            self.display.blit(self.font.render("Team 2", True, (255, 255, 255)), (130,310))
            pygame.draw.rect(self.display, (51,153,255), (100, 370, 150, 50), 2)
            s.fill(pygame.Color((51,153,255)), (100, 370, 150, 50))
            self.display.blit(self.font.render("Player 1", True, (255, 255, 255)), (130,380))
            pygame.draw.rect(self.display, (51,153,255), (100, 440, 150, 50), 2)
            s.fill(pygame.Color((51,153,255)), (100, 440, 150, 50))
            self.display.blit(self.font.render("Player 2", True, (255, 255, 255)), (130,450))

            # display input boxes
            pygame.draw.rect(self.display, (51,153,255), player_1_input)
            pygame.draw.rect(self.display, (51,153,255), player_2_input)
            pygame.draw.rect(self.display, (51,153,255), player_3_input)
            pygame.draw.rect(self.display, (51,153,255), player_4_input)
            self.display.blit(self.font.render(player_1_name, True, (255, 255, 255)), (310,130))
            self.display.blit(self.font.render(player_2_name, True, (255, 255, 255)), (310,200))
            self.display.blit(self.font.render(player_3_name, True, (255, 255, 255)), (310,380))
            self.display.blit(self.font.render(player_4_name, True, (255, 255, 255)), (310,450))

            #display start button
            pygame.draw.rect(self.display, (0,204,0), start_button)
            self.display.blit(self.font.render(start_button_text, True, (255, 255, 255)), (710,270))
        
            # always draw the mouse last 
            pos = pygame.mouse.get_pos()
            self.display.blit(self.mouse, (pos[0], pos[1] - self.mouse_scale_height))
            pygame.display.update()        
        
        #pygame.quit()
   
    
    def define_winner(self):
        highest_score = -1
        winner = ""
        for key in self.current_score.keys():
            if self.current_score[key] > highest_score:
                highest_score = self.current_score[key]
                winner = key

        return winner
    

        



if __name__ == "__main__":
    new_game = Game()
    new_game.run()
    