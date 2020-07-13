# DO NOT CHANGE THIS FILE
import nltk
from nltk.grammar import FeatureGrammar
from nltk.sem import cooper_storage as cs

from utils import display_latex, display_translation, display_tree, display, Markdown
from copy import deepcopy
from itertools import zip_longest

fcfg_string_notv = r"""
% start S
############################
# Grammar Rules
#############################

S[SEM = <?subj(?vp)>] -> NP[NUM=?n,SEM=?subj] VP[NUM=?n,SEM=?vp]

NP[NUM=?n,SEM=<?det(?nom)> ] -> Det[NUM=?n,SEM=?det]  Nom[NUM=?n,SEM=?nom]
NP[NUM=?n,SEM=?np] -> PropN[NUM=?n,SEM=?np]

Nom[NUM=?n,SEM=?nom] -> N[NUM=?n,SEM=?nom]

VP[NUM=?n,SEM=?v] -> IV[NUM=?n,SEM=?v]

PP[+TO, SEM=?np] -> P[+TO] NP[SEM=?np]

#############################
# Lexical Rules
#############################

PropN[NUM=sg,SEM=<\P.P(napoleon)>] -> 'Napoleon'
PropN[NUM=sg,SEM=<\P.P(moscow)>] -> 'Moscow'
PropN[NUM=sg,SEM=<\P.P(russia)>] -> 'Russia'
 
Det[NUM=sg,SEM=<\P Q.all x.(P(x) -> Q(x))>] -> 'every'
Det[NUM=pl,SEM=<\P Q.all x.(P(x) -> Q(x))>] -> 'all'
Det[NUM=sg,SEM=<\P Q.exists x.(P(x) & Q(x))>] -> 'a'
Det[NUM=sg,SEM=<\P Q.exists x.(P(x) & Q(x))>] -> 'an'

N[NUM=sg,SEM=<\x.man(x)>] -> 'man'
N[NUM=sg,SEM=<\x.bone(x)>] -> 'bone'
N[NUM=sg,SEM=<\x.dog(x)>] -> 'dog'
N[NUM=pl,SEM=<\x.dog(x)>] -> 'dogs'

P[+to] -> 'to'

""" 
fcfg_string_tv = """
#############################
# Grammar of transitive verbs and their lexical rules
#############################

VP[NUM=?n,SEM=<?v(?obj)>] -> TV[NUM=?n,SEM=?v] NP[SEM=?obj]
VP[NUM=?n,SEM=<?v(?obj,?pp)>] -> DTV[NUM=?n,SEM=?v] NP[SEM=?obj] PP[+TO,SEM=?pp]

TV[NUM=sg,SEM=<\X x.X(\y.bite(x,y))>,TNS=pres] -> 'bites'
TV[NUM=pl,SEM=<\X x.X(\y.bite(x,y))>,TNS=pres] -> 'bite'
DTV[NUM=sg,SEM=<\Y X x.X(\z.Y(\y.give(x,y,z)))>,TNS=pres] -> 'gives'
DTV[NUM=pl,SEM=<\Y X x.X(\z.Y(\y.give(x,y,z)))>,TNS=pres] -> 'give'

"""
syntax_notv = FeatureGrammar.fromstring(fcfg_string_notv)
syntax = FeatureGrammar.fromstring(fcfg_string_notv + fcfg_string_tv)

# don't change these functions
def sem_parser(sents, syntax, verbose=False, is_cs=False):
    """
    It parses sentences with an FDFG grammar and returns a dictionary of 
    sentences to their semantic representations.
    
    Parameters:
    sents: a list of sentences to be parsed.
    fcfg_string: a string listing all fcfg rules with SEM for the 
                 FeatureGrammar.
    verbose: boolean value. default value is `False`. 
             if verbose is True it prints results.
    is_cs: boolean value. Inicating if it is using Cooper Storage. Default value is `False`. 
    Returns:
    dict: dictionary of sentences translated to a list of their 
          semantic representaions.
    """
    sents_reps = {sent: [] for sent in sents}
    for sent, results in zip(sents, nltk.interpret_sents(sents, syntax)):
        if verbose:
            display(Markdown(f"----\n{sent}: {len(results)} result(s)"))
        for j, (synrep, semrep) in enumerate(results):
            if is_cs:
                cs_semrep = cs.CooperStore(semrep)
                cs_semrep.s_retrieve(trace=False)
                for reading in cs_semrep.readings:
                    sents_reps[sent].append(reading)
                    if verbose:
                        display_latex(reading) # prints the SEM feature of a tree
                if verbose:
                    display_tree(synrep) # show the parse tree
            else:
                sents_reps[sent].append(semrep)
                if verbose:
                    display_latex(semrep) # prints the SEM feature of a tree
                    display_tree(synrep) # show the parse tree
    return sents_reps


def evaluate_sentences(sents_reps, world):
    """
    Evaluates representation of each sentences in the world model.
    It translates them to their values: True or False.
    
    Parameters:
    sents_reps: dictionary of sentences to list of semantic representations.
    world: string that represents entities and sets of relations.
    
    Returns:
    dict: a dictionary of sentences to dictionary of semantic representations to values.
    """
    val = nltk.Valuation.fromstring(world)
    g = nltk.Assignment(val.domain)
    m = nltk.Model(val.domain, val)
    
    sents_reps = {
        sent: {
            str(semrep): m.evaluate(str(semrep), g)
            for semrep in sents_reps[sent]
        }
        for sent in sents_reps
    }
    
    return sents_reps

def compare_synsem(grammar1, grammar2):
    def syntax_strip(rule):
        return (list(rule.lhs().values())[0], " ".join(list(r.values())[0] if type(r) != str else '{{STR}}' for r in rule.rhs()))
    
    syntactic_categories = {
        syntax_strip(rule)
        for rules in [grammar1.productions(), grammar2.productions()]
        for rule in rules
    }

    syn_sem = {
        (cat, sem1, sem2)
        for cat in syntactic_categories
        for sem1, sem2 in zip_longest([
            rule.lhs()['SEM'] 
            for rule in grammar1.productions()
            if 'SEM' in rule.lhs()
            if syntax_strip(rule) == cat
        ],[
            rule.lhs()['SEM']
            for rule in grammar2.productions()
            if 'SEM' in rule.lhs()
            if syntax_strip(rule) == cat
        ])
    }
    
    syn_sem = sorted(list(syn_sem), key=lambda x: x[0])
    
    return syn_sem