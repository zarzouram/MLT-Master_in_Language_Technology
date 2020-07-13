## lookup:alphabet
data/alphabit.txt

## intent:choose_a_player
- I will go for [0](numerical)
- I will go for [1](numerical)
- I choose [0](numerical)
- I choose [1](numerical)
- [0](numerical)
- [1](numerical)
<!-- - one
- zero
- I choose one
- I go for one
- I choose zero
- I go for zero -->

## intent:play_ghost
- [A](alphabet)
- [a](alphabet)
- [B](alphabet)
- [b](alphabet)
- [C](alphabet)
- [c](alphabet)
- [J](alphabet)
- [j](alphabet)
- [k](alphabet)
- [K](alphabet)
- [m](alphabet)
- [O](alphabet)
- [o](alphabet)
- [T](alphabet)
- [t](alphabet)
- [u](alphabet)
- [Z](alphabet)
- [z](alphabet)

<!-- ## intent:inform
- [0](numerical)
- [1](numerical) -->
<!-- - [A](alphabet)
- [a](alphabet)
- [B](alphabet)
- [b](alphabet)
- [D](alphabet)
- [d](alphabet)
- [H](alphabet)
- [h](alphabet)
- [k](alphabet)
- [K](alphabet)
- [l](alphabet)
- [Q](alphabet)
- [q](alphabet)
- [T](alphabet)
- [t](alphabet)
- [u](alphabet)
- [Z](alphabet)
- [z](alphabet) -->

## regex:numerical
- \b\d{1}\b$

## regex:alphabet
- \b[a-zA-Z]{1}\b$

## intent:greet
- hey
- hello
- hi
- good morning
- good evening
- hey there

## intent:goodbye
- bye
- goodbye
- see you around
- see you later

## intent:affirm
- yes
- indeed
- of course
- that sounds good
- correct

## intent:deny
- no
- never
- I don't think so
- don't like that
- no way
- not really