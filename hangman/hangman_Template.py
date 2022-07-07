'''
Make sure you complete all the TODOs in this file.
The prints have to contain the same text as indicated, don't add any more prints,
or you will get 0 for this assignment.
'''
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
    def __init__(self, word_list, num_lives=5):
        # TODO 2: Initialize the attributes as indicated in the docstring
        # TODO 2: Print two message upon initialization:
        # 1. "The mystery word has {len(self.word)} characters" (The number of letters is NOT the UNIQUE number of letters)
        # 2. {word_guessed}
        
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
        # TODO 3: Check if the letter is in the word. TIP: You can use the lower() method to convert the letter to lowercase
        # TODO 3: If the letter is in the word, replace the '_' in the word_guessed list with the letter
        # TODO 3: If the letter is in the word, the number of UNIQUE letters in the word that have not been guessed yet has to be reduced by 1
        # TODO 3: If the letter is not in the word, reduce the number of lives by 1
        # Be careful! A word can contain the same letter more than once. TIP: Take a look at the index() method in the string class

        #compensate for different input casing by converting to lower case 
        letter = letter.lower()

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
        # TODO 1: Ask the user for a letter iteratively until the user enters a valid letter
        # TODO 1: Assign the letter to a variable called `letter`
        # TODO 1: The letter has to comply with the following criteria: It has to be a single character. If it is not, print "Please, enter just one character"
        # TODO 2. It has to be a letter that has not been tried yet. Use the list_letters attribute to check this. If it has been tried, print "{letter} was already tried".
        # TODO 3: If the letter is valid, call the check_letter method


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
                            
        

def play_game(word_list):
    # As an aid, part of the code is already provided:
    game = Hangman(word_list, num_lives=5)
    # TODO 1: To test this task, you can call the ask_letter method
    # TODO 2: To test this task, upon initialization, two messages should be printed 
    # TODO 3: To test this task, you call the ask_letter method and check if the letter is in the word
    
    # TODO 4: Iteratively ask the user for a letter until the user guesses the word or runs out of lives
    # If the user guesses the word, print "Congratulations, you won!"
    # If the user runs out of lives, print "You ran out of lives. The word was {word}"
    
    ##call class and method
    game.ask_letter()
    

if __name__ == '__main__':
    word_list = ['apple', 'banana', 'orange', 'pear', 'strawberry', 'watermelon']
    play_game(word_list)
# %%
