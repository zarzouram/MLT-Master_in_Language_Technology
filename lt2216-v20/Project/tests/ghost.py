# ghost
# see more at http://en.wikipedia.org/wiki/Ghost_(game)

import random

minimum_length = 5

# create a list of the supplied word list
wordlistpath = '/usr/share/dict/words'
with open(wordlistpath, "r") as f:
    wordlist = f.readlines()

words = [word.rstrip().lower()
         for word in wordlist if len(word.rstrip()) >= minimum_length]
possible_words = words

current_player = random.choice([0, 1])
current_root = ''
current_choice = ''
possible_choices = ''

which_player = lambda p: p and 'bot' or 'human'

print(which_player(current_player), 'starts')

# game loop
while True:
    # find the possible words given the current root
    possible_words = [
        word for word in possible_words if word[:len(current_root)] == current_root]
    # find the legal letter choices available to the next player
    possible_choices = ''
    # truncate the current root from the possible words and pull the first letter
    for letter in [word[len(current_root):][0] for word in possible_words]:
        if letter not in possible_choices:
            possible_choices += letter

    # let the next player make a choice
    if current_player == 0:
        current_choice = str(input('u pick: '))
        # check if the choice is legal
        if current_choice not in possible_choices:
            print('illegal choice, %s loses' % which_player(current_player))
            break
    elif current_player == 1:
        current_choice = possible_choices[random.randint(
            0, len(possible_choices) - 1)]
        print('i pick:', current_choice)

    current_root += current_choice
    print(current_root)

    # check whether a word has been completed
    if current_root in words:
        print('completed word, %s loses' % which_player(current_player))
        break

    # rotate player
    if current_player == 1:
        current_player = 0
    else:
        current_player = 1
