import cv2
from keras.models import load_model
import numpy as np
import random
import time

'''
Create two functions: get_computer_choice and get_user_choice. 
The first function will randomly pick an option between "Rock", "Paper", and "Scissors" and return the choice.
The second function will ask the user for an input and return it.

the model now has a nothing in its ranking compared to the manaul rps

#todo when there is a draw it only reports one over winner. Fix so it says it is a total draw.

'''



class rps:


    def __init__(self, game_choice):

        #take the dict of choices and bind it to the class
        self.game_choice = game_choice

        ##get the options for the game it will be used for the computer and user
        self.computer_options = list(self.game_choice.keys())
        ##swap the keys around for later to work out the winner
        self.math_choice = dict([(value, key) for key, value in self.game_choice.items()])

        ##initalise the rest of code 
        self.user_choice=()
        self.overall_winner={}


    def get_computer_choice(self):
        
            #choose one of the three 
            comp_chosen= random.choice(self.computer_options )
            
            ##get numerical rank
            self.computer_chosen_rank =  self.game_choice[comp_chosen.lower()]

            #update the dict so when competition starts we know who played what 
            self.math_choice.update({self.computer_chosen_rank : 'the AI'})
            #print (self.computer_chosen_rank)
        

    def get_user_choice(self):        

        #take in the user input
        self.user_choice = input("Enter rock or paper or scissors: ").lower()
        #print (self.user_choice)
        #print (self.computer_options)
        #make sure input is valid
        self.check_user_input()    

        #get the numerical rank 
        self.user_chosen_rank = self.game_choice[self.user_choice]

        #update the dict so when competition starts we know who played what
        self.math_choice.update({self.user_chosen_rank : 'you'})
                
     

    def get_winner(self):

        '''
        Using if-elif-else statements
        Wrap the code in a function called get_winner and return the winner. This function takes two arguments: computer_choice and user_choice.
        '''        
        
        if self.computer_chosen_rank == self.user_chosen_rank:
            ##can use either the user choice or comp choice to return
            print (f"This is a draw both you and computer played a {self.user_choice}\n")
        elif (self.computer_chosen_rank - self.user_chosen_rank)**2 == 1:
            ##when consecutive number work out the max value
            largest_rank = max(self.computer_chosen_rank, self.user_chosen_rank)
            print (f"Winner is {self.math_choice[largest_rank]}\n")
            ###increment the counter to check who the overall winner is.
            self.overall_winner[self.math_choice[largest_rank]] =  self.overall_winner.get(self.math_choice[largest_rank],0) +1 
        else:
            # non consecutive number therefore min number is the winner
            smallest_rank = min(self.computer_chosen_rank, self.user_chosen_rank)
            print (f"Winner is {self.math_choice[smallest_rank]}\n")
            ##increment the counter to check who the overall winner is.
            self.overall_winner[self.math_choice[smallest_rank]] =  self.overall_winner.get(self.math_choice[smallest_rank],0) +1
               
        
    def check_user_input(self):
        
        ###check valid input

        if self.user_choice not in  self.computer_options:
            print (f'This is not a valid input. You can only input one of these three choices '+ ','.join(self.computer_options))       
            exit()


    def get_prediction(self):
        ##load the model from the teachable machine. 
        model = load_model('keras_model.h5')
        ##use camera to capture the image
        cap = cv2.VideoCapture(0)
        ##create container to hold data
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        stop_playing = time.time() + 5
        while time.time() < stop_playing: 
            ##take input from camera image
            ret, frame = cap.read()
            ###make sure image is scaled down - note same as container/array
            resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            image_np = np.array(resized_frame)
            normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
            data[0] = normalized_image
            prediction = model.predict(data)

            ##add counter to image
            # Display countdown on each frame
            # specify the font and draw the
            # countdown using puttext
            #format string to make it look nice in image
            counter = ", ".join(f"{key} {value}" for key, value in self.overall_winner.items())
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(counter),
                        (20, 100), font,
                        1, (0, 255, 255),
                        4, cv2.LINE_AA)



            ##display the image in the frame
            cv2.imshow('frame', frame)
            # Press q to close the window
            print(prediction) 
            ###get the prediction with the highest confidence score
            ###np provides argmax
            self.user_chosen_rank = np.argmax(prediction)
            #update the dict so when competition starts we know who played what
            self.math_choice.update({self.user_chosen_rank : 'you'})
            ##who is the winner
            self.get_winner()

            #check values to see if they equal 3. Time constraint still in place
            if 28 in self.overall_winner.values():
                break
            
            
            #this corresponds to the matrix below. Kill with q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # After the loop release the cap object
        cap.release()
        # Destroy all the windows
        cv2.destroyAllWindows()



def play_game(game_choice):

    ##call the class
    gameon = rps(game_choice)
    x =1
    #while x in range(1,5):
    #gameon = rps(game_choice)
    gameon.get_computer_choice()
    #gameon.get_user_choice()
    gameon.get_prediction()
    ##start the competition
    #gameon.get_winner()
    print("Next round\n")
    #    x +=1
    
    ###Provide a report at the end of the final scores
    print (f"#The final results are;\n")
   
    for player,value in gameon.overall_winner.items():
        ##make it easier for the eye
        player = player.title()
        print(f"{player} won {value}")

    ##limitation is both players have equal wins
    overall_winner = max(gameon.overall_winner, key=gameon.overall_winner.get) 
    print (f"The overall winner is {overall_winner}\n")

    print ('END')




if __name__ == '__main__':
    game_choice = {'rock':0, 'paper':1, 'scissors':2}
    play_game(game_choice)


