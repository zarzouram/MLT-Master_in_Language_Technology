import email
import glob
from os.path import basename, dirname
# import talon
# talon.init()
# from talon import signature
import re

subjectptrn = r"([\[\(] *)?(RE|FWD?) *([-:;)\]][ :;\])-]*|$)|\]+ *$"
signatureptrn = r"(^(.|(\w*\s)){0,2}[r|R]egard(s?),?$)|(^\s*\*+).*?This.*?contain.*?(confidential|privilege|proprietary).*?sender"
furthercleaningptrn = r"(^>*\s*-+) *Forwarded by.*?-+.*?Subject:"
replyptrn = r"(^\s*>+)(\s*>*)*"
spacesptrn = r"^ +| +$"
regex = re.compile(subjectptrn, flags = re.I | re.M)
spacesre = re.compile(spacesptrn, flags = re.I | re.M)
resignature = re.compile(signatureptrn, flags = re.I | re.M| re.S)
furthercleaningre = re.compile(furthercleaningptrn, flags = re.I | re.M| re.S)
replyre = re.compile(replyptrn, flags = re.I | re.M)


def check_emails_multipart(filepath):
    with open(filepath, "r", encoding="utf-8") as fin:
        theemail = email.message_from_file(fin)
    
    if theemail.is_multipart():
        yield filepath
    else:
        yield None

# def get_data(docpath):
#     filespath = glob.iglob("{}/*".format(docpath))
#     docdata = []
#     for filepath in filespath:
#         docdata.append(get_email_data(filepath))

#     return {basename(docpath): docdata}

def remove_signature(text, emailsender=None):
    # mytext, _ = signature.extract(
    #     text, sender=emailsender)
    mytext = re.split(resignature, text, 1)[0]

    return mytext


def get_data(filepath):
    '''Read an email file, remove header, extract email body text from the main email and from the messeges that are replied to/forwarded.
    '''
    with open(filepath, "r") as f:
        theemail = email.message_from_file(f)

    # extract body text
    payload = theemail.get_payload(decode=True)
    bodytext = payload.decode()
    bodytext = re.sub(replyre, "", bodytext)
    bodytext = re.sub(spacesre, "", bodytext)
    # extract Subject Line
    subject = theemail['subject']
    subject = re.sub(regex, '', subject) if subject else ""
    
    # check if there are othere emails in email, remove signature and extract body messege only.
    if "-----Original Message-----\n" in bodytext:
        bodiestext = []
        emailssplitted = bodytext.split("-----Original Message-----\n")
        # remove signature
        # thesender = theemail['from']
        mytext = remove_signature(emailssplitted[0])
        mytext = re.sub(furthercleaningre, "", mytext)
        bodiestext.append(mytext)
        if subject != "": bodiestext.append(subject)

        for i in range(1, len(emailssplitted)):
            # extraxt body message
            # mymsg = .lstrip()
            originalemail = email.message_from_string(emailssplitted[i])
            payload = originalemail.get_payload(decode=True)
            bodytext = payload.decode()
            # remove signature
            # thesender = originalemail['from']
            bodytext = remove_signature(bodytext)
            bodytext = re.sub(furthercleaningre, "", bodytext)
            bodiestext.append(bodytext)
            # extract subject
            subject = originalemail['subject']
            subject = re.sub(regex, '', subject) if subject else ""
            if subject != "": bodiestext.append(subject)
        
        bodytext = " ".join(bodiestext)
    else:
        # remove signature
        # thesender = theemail['from']
        bodytext = remove_signature(bodytext)
        bodytext = re.sub(replyre, "", bodytext)
        bodytext = re.sub(furthercleaningre, "", bodytext)
        bodytext += " " + subject

    bodytext = re.sub(r"\s+", " ", bodytext)
    bodytext = bodytext.strip()
    # return sender and body text
    return [bodytext, basename(dirname(filepath))]

if __name__ == "__main__":
    enrondir = "/home/guszarzmo@GU.GU.SE/Corpora/enron-emails/enron_sample/"
    enronfilespath = glob.iglob(enrondir+"*/*.")
    test = []
    for msgpath in enronfilespath:
        for flag in get_data(msgpath):
            test.append(1)
    u=1
