import random

from numpy import compare_chararrays

'''
Create two functions: get_computer_choice and get_user_choice. 
The first function will randomly pick an option between "Rock", "Paper", and "Scissors" and return the choice.
The second function will ask the user for an input and return it.

'''



class rps:


    def __init__(self, game_choice):

        self.game_choice = game_choice

        ###assign numbers

        self.computer_options = list(self.game_choice.keys())
        ##swap the keys around for later to work out the winner
        self.math_choice = dict([(value, key) for key, value in self.game_choice.items()])
        print (self.game_choice)
        print (type(self.game_choice))
        print (self.computer_options)

        ##initalise the rest of code 
        self.user_choice=()


    def get_computer_choice(self):
        
            #choose one of the three 
            comp_chosen= random.choice(self.computer_options )
            ##get numerical rank
            self.computer_chosen_rank =  self.game_choice[comp_chosen.lower()]

            #update the dict so when competition starts we know who played what 
            self.math_choice.update({self.computer_chosen_rank : 'the AI'})
            print (self.computer_chosen_rank)
        

    def get_user_choice(self):

        

        #take in the user input
        self.user_choice = input("Enter rock or paper or scissors ").lower()
        #print (self.user_choice)
        #print (self.computer_options)

        #get the numerical rank 
        self.user_chosen_rank = self.game_choice[self.user_choice]

        #update the dict so when competition starts we know who played what
        self.math_choice.update({self.user_chosen_rank : 'you'})
        #make sure input is valid
        self.check_user_input()            
     

    def get_winner(self):

        '''
        Using if-elif-else statements
        Wrap the code in a function called get_winner and return the winner. This function takes two arguments: computer_choice and user_choice.
        '''        
       
        if self.computer_chosen_rank == self.user_chosen_rank:
            ##can use either the user choice or comp choice to return
            print (f'This is a draw both you and computer played a {self.user_choice}')
        elif (self.computer_chosen_rank - self.user_chosen_rank)**2 == 1:
            ##when consecutive number work out the max value
            largest_rank = max(self.computer_chosen_rank, self.user_chosen_rank)
            print (f'Winner is {self.math_choice[largest_rank]}')
        else:
            # non consecutive number therefore min number is the winner
            smallest_rank = min(self.computer_chosen_rank, self.user_chosen_rank)
            print (f'Winner is {self.math_choice[smallest_rank]}')
               
        
    def check_user_input(self):
        
        ###check valid input

        if self.user_choice not in  self.computer_options:
            print (f'This is not a valid input. You can only input one of these three choices '+ ','.join(self.computer_options))       
            exit()



def play_game(game_choice):

    gameon = rps(game_choice)

    while range(1,5):

        gameon.get_computer_choice()
        gameon.get_user_choice()
        ##start the competition
        gameon.get_winner()
    









if __name__ == '__main__':
    game_choice = {'rock':0, 'paper':1, 'scissors':3}
    play_game(game_choice)