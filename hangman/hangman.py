import random

class Hangman:
    '''
    A Hangman Game that asks the user for a letter and checks if it is in the word.
    It starts with a default number of lives and a random word from the word_list.

    
    Parameters:
    ----------
    word_list: list
        List of words to be used in the game
    num_lives: int
        Number of lives the player has
    
    Attributes:
    ----------
    word: str
        The word to be guessed picked randomly from the word_list
    word_guessed: list
        A list of the letters of the word, with '_' for each letter not yet guessed
        For example, if the word is 'apple', the word_guessed list would be ['_', '_', '_', '_', '_']
        If the player guesses 'a', the list would be ['a', '_', '_', '_', '_']
    num_letters: int
        The number of UNIQUE letters in the word that have not been guessed yet
    num_lives: int
        The number of lives the player has
    list_letters: list
        A list of the letters that have already been tried

    Methods:
    -------
    check_letter(letter)
        Checks if the letter is in the word.
    ask_letter()
        Asks the user for a letter.
    '''
    def __init__(self, word_list, num_lives=6):
        ##hangmand requires 6 
        ##select a random word from the list
        self.word = random.choice(word_list).lower()
        ##print out the result
        print(f'The mystery word has {len(self.word)} characters')

        ##create placeholder list for letters inputted  
        self.word_guessed = ["_"] * len(self.word)
        print (f'The word to be guessed is {self.word_guessed}')
        
        ##initialise the rest of the attributes.
        self.num_letters = len(set(self.word))
        self.num_lives = num_lives
        self.list_letters =[]    
        
                

    def check_letter(self, letter) -> None:
        '''
        Checks if the letter is in the word.
        If it is, it replaces the '_' in the word_guessed list with the letter.
        If it is not, it reduces the number of lives by 1.

        Parameters:
        ----------
        letter: str
            The letter to be checked

        '''
        #compensate for different input casing by converting to lower case 
        letter = letter.lower()

        #get the graphic dictionary
        self.graphic()

        ##Capture the letter used
        self.list_letters.append(letter)
        
         ### is this a valid letter and has this been tried before
        if letter in self.word:            

            ##reduce the count for unique letters
            self.num_letters -= 1

            ##get all the indexes and their character then check that the character equals the inputted letter
            ##and populate relevant index value to letter_indexes, to replace them later. 
            letter_indexes = [i for i, c in enumerate(self.word) if c == letter]
                        
            ##Looping of indexes compensates for more than one instance of a character 
            for i in letter_indexes:
           
                #simple method to overwrite index in word 
                self.word_guessed[i] = letter
                
            #check it works also user feedback
            print(self.word_guessed)

            ##if all underscores gone the word is complete. WIN
            if '_' not in self.word_guessed:
                print ("Congratulations you have won")
                exit()    
            
            ##now reduce counter of self.num_letters
            self.num_letters -= 1
        else:
            ##two scenarios; letter has been guessed before and it is the incorrect letter
            print (f'Letter: "{letter}" is not in the word')
            print (f'{self.hang_graphic[self.num_lives]}') 
            self.num_lives -= 1
            

            ##run out of lives if value is zero
            if self.num_lives == 0:
                print ("You have run out of lives.")
                exit()

        

    def ask_letter(self):
        '''
        Asks the user for a letter and checks two things:
        1. If the letter has already been tried
        2. If the character is a single character
        If it passes both checks, it calls the check_letter method.
        '''
      
        while True:

            letter = input("Enter letter: ")
            print ("Letter entered is:", letter)

            if len(letter) > 1:
                print ("Please, enter just one character")
                False  
            elif letter in self.list_letters:
                print("Please try another letter you have already tried it")
                False                
            else:
                self.check_letter(letter) 

    def graphic(self):

        self.hang_graphic = {}

        self.hang_graphic = {
            5:'\
                __________\n\
               |          |\n\
               |         (O)\n\
               |\n\
               |\n\
               |\n\
               |\n\
               |\n\
            ___|___ \n',
            
            4:'\
                __________\n\
               |          |\n\
               |         (O)\n\
               |          |\n\
               |          |\n\
               |\n\
               |\n\
               |\n\
            ___|___ ',
            
            3:'\
                __________\n\
               |          |\n\
               |         (O)\n\
               |          |\n\
               |          |\n\
               |         / \n\
               |        / \n\
               |\n\
            ___|___ ',

            2:'\
                __________\n\
               |          |\n\
               |         (O)\n\
               |          |\n\
               |          | \n\
               |         / \  \n\
               |        /   \  \n\
               |\n\
            ___|___ ',
            1:'\
                __________\n\
               |          |\n\
               |         (O)\n\
               |         _|\n\
               |        / |\n\
               |         / \ \n\
               |        /   \\n\
               |\
            ___|___ ',
            0:'\
                __________\n\
               |          |\n\
               |         (O)\n\
               |         _|_\n\
               |        / | \ \n\
               |         / \ \n\
               |        /   \ \n\
               |      Game Over\n\
            ___|___ ',
        }

        
        

def play_game(word_list):
    # As an aid, part of the code is already provided:
    game = Hangman(word_list, num_lives=5)    
    ##call class and method
    game.ask_letter()
    

if __name__ == '__main__':
    word_list = ['apple', 'banana', 'orange', 'pear', 'strawberry', 'watermelon']
    play_game(word_list)
# %%
