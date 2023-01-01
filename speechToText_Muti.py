
import os
import ssl
import sys
import click
import base64
import  requests
import  nltk
import string
import whisper
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Levenshtein import distance as lev
import nltk.data
import nltk
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method4
nato_tune=True
nato_list = ['ALPHA', 'BRAVO', 'CHARLIE', 'DELTA', 'ECHO', 'FOXTROT', 'GOLF', 'HOTEL', 'INDIA', 'JULIET', 'KILO',
             'LIMA',
             'MIKE', 'NOVEMBER', 'OSCAR', 'PAPA', 'QUEBEC', 'ROMEO', 'SIERRA', 'TANGO', 'UNIFORM', 'VICTOR',
             'WHISKEY',
             'X-RAY', 'YANKEE', 'ZULU', "Recording license plate", "Reporting license plate",
             "recording license plate", "reporting license plate"]
nato_dict = {'alpha': 'a', 'bravo': 'b', 'charlie':'c', 'delta': 'd', 'echo':'e', 'foxtrot':'f', 'golf':'g',
 'hotel':'h', 'india':'i', 'juliet':'j', 'kilo':'k', 'lima':'l', 'mike':'m', 'november':'m', 'oscar':'o',
'papa': 'p', 'quebec': 'q', 'romeo': 'r', 'sierra':'r', 'tango':'t', 'uniform':'u', 'victor':'v', 'whiskey':'w',
'x-ray':'x', 'yankee':'y', 'zulu':'z'}
class SpeechToTextEngine:
    # def __init__(self):
    #     # raise Exception("Not implemented")

    def transcribe(self, wav):
        raise Exception("Not implemented")
    def get_exp_run_jan(self, expid):
        # def getjsonresponse(expid):
        api_url = "https://mcv-testbed.cs.columbia.edu/api/experiment_run/" + expid
        response = requests.get(api_url)
        if response.status_code == 200:
            print("Good api to go ")
            # using wave to text
            # resp=json.loads(json.dump(response))
            resp = response.json()
            # print( resp['audio'])
            return resp, resp['experiment']
        else:
            # raise exception
            raise Exception('Wrong api expid id ')



class GoogleSTT(SpeechToTextEngine):
    def __init__(self, opts):
        if opts.get('key', None) is None:
            raise Exception('Missing Google STT key')
        self.key = opts['key']
    def get_exp_run_answer(self, expid):
        api_url = "https://mcv-testbed.cs.columbia.edu/api/experiment/" + expid
        response = requests.get(api_url)
        correct_array = []
        if response.status_code == 200:
            print("Good api to go answer ")
            # using wave to text
            # resp=json.loads(json.dump(response))
            resp = response.json()
            for item in resp["steps"]:
                correct_array.append(str(item["correct_answer"]).lower())
                # print(item["correct_answer"])
            print(correct_array)
        return correct_array

    def get_exp_run(self, expid):
        # def getjsonresponse(expid):
        api_url = "https://mcv-testbed.cs.columbia.edu/api/experiment_run/" + expid
        response = requests.get(api_url)
        if response.status_code == 200:
            print("Good api to go ")
            # using wave to text
            # resp=json.loads(json.dump(response))
            resp = response.json()
            # print( resp['audio'])
            return resp['audio']
        else:
            # raise exception
            raise Exception('Wrong api expid id ')
    def get_exp_run_jan(self, expid):
        # def getjsonresponse(expid):
        api_url = "https://mcv-testbed.cs.columbia.edu/api/experiment_run/" + expid
        response = requests.get(api_url)
        if response.status_code == 200:
            print("Good api to go ")
            # using wave to text
            # resp=json.loads(json.dump(response))
            resp = response.json()
            print(  resp["experiment"])
            return resp, resp["experiment"]
        else:
            # raise exception
            raise Exception('Wrong api expid id ')

    def getbinaryitemtobased64( self , audio_cur ):
        resp = requests.get(audio_cur)
        return base64.b64encode(resp.content).decode('utf-8')

    def getresponsebyapiurl(self, cur_64):
        API_URL = "https://speech.googleapis.com/v1p1beta1/speech:recognize?key=" + self.key
        global nato_tune
        # print(f"Sending {items} to Google")
        # # doesn't matter if truncated: will know to

        post_request = {
            "config": {
                "encoding": "LINEAR16",
                "languageCode": "en-US",

            },

            "audio": {
                "content": cur_64,
            }
        }
        if nato_tune:
            post_request["config"]["speechContexts"] = [{
                "phrases": [nato_list],
                "boost": 100
            }]
        request = requests.post(API_URL, json=post_request)
        data = request.json()
        # print(request)
        try:
            # if request.status_code == 200:
            #     print(data['results'][0]['alternatives'][0]['transcript'])
            #     # print(data)
            plaintext = data['results'][0]['alternatives'][0]['transcript']
            pure_text = str(plaintext).translate(str.maketrans('', '', string.punctuation))
            print("---------------------------------")
            return pure_text
            # else:
            #     raise Exception('Wrong api expid id ')
        except KeyError:
            # raise Exception("Key error ")
            # print(f"Key error caused by {data}")
            # return 0
            print("There is a 15s check for api_ url ")
            return ""
    def transcribe(self,wav):
        current_audio = GoogleSTT.getbinaryitemtobased64(self, wav)
        cur_text = GoogleSTT.getresponsebyapiurl(self,current_audio)
        start = wav.find("impaired")
        name = wav[start:len(wav)]
        return name , cur_text
import whisper
import torch
class Whisper(SpeechToTextEngine):
    def transcribe(self, wav):
        # whisper.DecodingOptions(fp16=False)
        # backends.mps.is_available()
        # print(torch.backends.mps.is_available())
        device = torch.device("mps")
        device = torch.device("cuda:0")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model = whisper.load_model("base.en")
        model = whisper.load_model("medium.en")
        # model = whisper.load_model("large")
        # model = whisper.load_model("base.en")
        result = model.transcribe(wav)
        start = wav.find("impaired")
        name = wav[start:len(wav)]
        list_punct = list(string.punctuation)
        pure_text = result["text"].translate(str.maketrans('', '', string.punctuation))
        # print(result["text"])
        # print(pure_text)
        return name, pure_text
    def get_exp_run_jan(self, expid):
        api_url = "https://mcv-testbed.cs.columbia.edu/api/experiment_run/" + expid
        response = requests.get(api_url)
        if response.status_code == 200:
            print("Good api to go ")
            # using wave to text
            # resp=json.loads(json.dump(response))
            resp = response.json()
            # print( resp['audio'])

            return resp, resp['experiment']
    def get_exp_run_answer(self, expid):
        api_url = "https://mcv-testbed.cs.columbia.edu/api/experiment/" + expid
        response = requests.get(api_url)
        correct_array = []
        if response.status_code == 200:
            print("Good api to go answer ")
            # using wave to text
            # resp=json.loads(json.dump(response))
            resp = response.json()
            for item in resp["steps"]:
                correct_array.append(str(item["correct_answer"]).lower())
                # print(item["correct_answer"])
            print(correct_array)
        return correct_array

# @click.command()
# @click.option("-i", "--id", required=True, help="input an id")
# @click.option("-n", "--num", type=int, help="input  a number", show_default=True)
# def main(id, num):
#     click.echo(f"your {id=} {num=}")
#
# if __name__ == '__main__':
#     main()

# @click.command()
# @click.option('-e', '--engine', help='Speech-to-text engine', show_default=True)
# # @click.argument('exp_run_id', nargs=1)
# @click.option('-id', '--exp_run_id', show_default=True)
# @click.pass_context
def get_exp_run_answer(expid):
    api_url = "https://mcv-testbed.cs.columbia.edu/api/experiment/" + expid
    response = requests.get(api_url)
    correct_array = []
    if response.status_code == 200:
        print("Good api to go answer ")
        # using wave to text
        # resp=json.loads(json.dump(response))
        resp = response.json()
        for item in resp["steps"]:
            correct_array.append(str(item["correct_answer"]).lower())
            # print(item["correct_answer"])
        # print(correct_array)
    return correct_array
def cli(engine, exp_run_id):
    if engine == 'GoogleSTT':
        try:
            apikey = os.environ['GOOGLE_STT_KEY']
            # apikey = "AIzaSyB-5VKtWsx7yCGCOxHRfpRDyZGjU8f4N80"
            stt = GoogleSTT({'key': apikey})
        except KeyError:
            print("Please specify Google Speech-to-text key in GOOGLE_STT_KEY environment variable")
            sys.exit(1)
    elif engine == 'whisper':
        stt = Whisper()
        # download the run...
    # print(expr_run)
    expr_run, answer_id = stt.get_exp_run_jan(exp_run_id)
    expr_answer = stt.get_exp_run_answer(answer_id)
    answer_dict = {}
    for wav in expr_run['audio']:
        name, transcription = stt.transcribe(wav=wav)
        answer_dict[name] = transcription
    print(answer_dict)
    return answer_dict, expr_answer
def getsameitem(list1 , list2):
    find_liencse  = []
    for el in list1:
        for cur in list2:
            if el == cur :
                find_liencse.append(cur)
            else:
                continue
    return find_liencse
def getwordindict (ans_dict):
    stpwrd = nltk.corpus.stopwords.words('english')
    smoothie = SmoothingFunction().method4
    # stpwrd = []
    new_stopwords = ["reporting", "license", "plate", "putting", "recording", "Reporting", "Supporting", "Recording", "The", "License", "Plate", "life"]
    stpwrd.extend(new_stopwords)
    clear=[]
    pure =[]
    # ans=[]
    for key, item in ans_dict.items():
        text_tokens= word_tokenize(item)
        removing_custom_words = [words for words in text_tokens if not words in stpwrd]
        clear.append((removing_custom_words))
        # print(removing_custom_words)
        ans=[]
        for cur in removing_custom_words:
            # print(item)
            if cur.isdigit():
                ans.append(cur)
            elif cur.lower() in nato_dict.keys():
                ans.append(cur)
            elif cur.lower() not in nato_dict.keys():
                curcmax= -1
                confidence_dict={}
                for item in nato_dict.keys():
                    confidence_dict[item] = bleu([cur.lower()], item, smoothing_function=smoothie)
                for k, v in confidence_dict.items():
                    needed =  max(list(confidence_dict.values()))
                    # print(needed)
                    if needed == v:
                        ans.append(k)
                            # print(k)
        pure.append(ans)
    print(pure)
                            # print(v)
                        # ans.append(k)
    lic=[]
    for item in pure:
        answer_licence = str()
        for cur in item:
            if cur.isdigit():
                answer_licence += (cur)
            else:
                answer_licence += str(cur[0:1].lower())
        lic.append(answer_licence)
    return lic







def getwordlev (ans_dict):
    stpwrd = nltk.corpus.stopwords.words('english')
    smoothie = SmoothingFunction().method4
    # stpwrd = []
    new_stopwords = ["reporting", "license", "plate", "putting", "recording", "Reporting", "Supporting", "Recording", "The", "License", "Plate", "life"]
    stpwrd.extend(new_stopwords)
    clear=[]
    pure =[]
    # ans=[]
    for key, item in ans_dict.items():
        text_tokens= word_tokenize(item)
        removing_custom_words = [words for words in text_tokens if not words in stpwrd]
        clear.append((removing_custom_words))
        # print(removing_custom_words)
        ans=[]
        for cur in removing_custom_words:
            # print(item)
            if cur.isdigit():
                ans.append(cur)
            elif cur.lower() in nato_dict.keys():
                ans.append(cur)
            elif cur.lower() not in nato_dict.keys():
                curcmax= -1
                confidence_dict={}
                for item in nato_dict.keys():
                    score= lev(cur.lower(),item)
                    confidence_dict[item] = 1 - score / max(len(cur.lower()), len(item))
                for k, v in confidence_dict.items():
                    needed =  max(list(confidence_dict.values()))
                    # print(needed)
                    if needed == v:
                        ans.append(k)
                            # print(k)
        pure.append(ans)
    print(pure)
                            # print(v)
                        # ans.append(k)
    lic=[]
    for item in pure:
        answer_licence = str()
        for cur in item:
            if cur.isdigit():
                answer_licence += (cur)
            else:
                answer_licence += str(cur[0:1].lower())
        lic.append(answer_licence)
    return lic

    #     str_1= "alpha"
    # str_2= "vlpha"
    # score= lev("alpha", "vlpha")
    # print(score)
    # similarity = 1 - score / max(len(str_1), len(str_2))
    # print(similarity)

def nlp_getliencse (ans_dict):
    stpwrd = nltk.corpus.stopwords.words('english')
    # stpwrd = []
    new_stopwords = ["reporting", "license", "plate", "putting", "recording", "Reporting", "Supporting", "Recording", "The", "License", "Plate", "life"]
    stpwrd.extend(new_stopwords)
    clear=[]
    for key, item in ans_dict.items():
        text_tokens= word_tokenize(item)
        removing_custom_words = [words for words in text_tokens if not words in stpwrd]
        clear.append((removing_custom_words))
    license_list=[]
    license_dirty=[]
    count=0
    for removing_custom_words_cur in clear:
        answer_licence = str()
        answer_licence_dir = str()
        for cur in removing_custom_words_cur:
            if len(removing_custom_words_cur) > 3:
                if cur.isdigit():
                    answer_licence += cur
                    answer_licence_dir += cur
                elif cur.lower() in nato_dict.keys():
                    answer_licence += nato_dict[cur.lower()]
                    answer_licence_dir  += nato_dict[cur.lower()]
                else :
                    answer_licence += "$"+cur+"$"
                    answer_licence_dir += str(cur[0:1].lower())
        count = count +1
        license_list.insert( count, answer_licence)
        license_dirty.insert(count, answer_licence_dir)
    clear_license=[]
    while ("" in license_dirty):
        license_dirty.remove("")
    for i in license_list:
        if "$" not in i and i != '' :
            clear_license.append(i)
    return clear_license , license_dirty
    # dirty_license=[]
    # for i in license_dirty:
    #     c
if __name__ == '__main__':
    # cli( engine=sys.argv[1], exp_run_id=sys.argv[2])
    # expid = "634d6514ea4db11e1b3caff1"
    answer_dict, expr_answer=cli( engine=sys.argv[1], exp_run_id=sys.argv[2])

    clear_license,license_dirty= nlp_getliencse(answer_dict)
    # expr_answer = get_exp_run_answer(expid)
    print("see")
    print(expr_answer)


    # print(clear_license)
    # print(license_dirty)
    # getwordindictlex(answer_dict)

    # cur_rate=len(clear_license)/72
    # print("correct rate: "+str(cur_rate ))
    # dirty_rate=len(license_dirty)/72
    # print("dirty rate: " + str(dirty_rate))
    # # print(expr_answer)
    got_license= getsameitem(list(expr_answer), license_dirty)
    # print(got_license)


    correct_rate = len(got_license)/72
    print("correct rate: " + str(correct_rate))

    ans1 = getwordindict(answer_dict)
    print( "correct rate with Bleu : " + str ( len(getsameitem(expr_answer,ans1 ))/72))
    ans2 =   getwordlev(answer_dict)
    print( "correct rate with lex : " + str ( len(getsameitem(expr_answer,ans2 ))/72))
    combine=got_license
    combine.extend(ans1)
    print( "correct rate with combine: " + str ( len(getsameitem(expr_answer,ans2 ))/72))
    # lev_arr=[]
    # for i in clear_license:
    #     for j in expr_answer:
    #         # print(lev(i, j))
    #         if (lev(i, j) == 5) :
    #             lev_arr.append(i)
    # # print(len(lev_arr))
    # # print( lev_arr)
    # # need_improve_array= [ *set(lev_arr)]
    # need_improve_array = list(dict.fromkeys(lev_arr))
    # for i in need_improve_array:
    #     if i in got_license :
    #         need_improve_array.remove(i)
    # print(need_improve_array)

    # print(len(need_improve_array))




    # for i  in
    # print(answ_dict)


    # We can also use the json module in the case when dictionaries or some other data can be easily mapped to JSON format.

# import json
#
# # Serialize data into file:
# json.dump( data , open( "file_name.json", 'w' ) )
#
# # Read data from file:
# data = json.load( open( "file_name.json" ) )
# GOOGLE_STT_KEY=AIzaSyB-5VKtWsx7yCGCOxHRfpRDyZGjU8f4N80 python speechToText_Muti.py GoogleSTT 61fcaa1f2fc54df92426936d
    # GOOGLE_STT_KEY=AIzaSyB-5VKtWsx7yCGCOxHRfpRDyZGjU8f4N80 python speechToText_Muti.py GoogleSTT 62befe7c27977f737525c4c3
    # GOOGLE_STT_KEY=AIzaSyB-5VKtWsx7yCGCOxHRfpRDyZGjU8f4N80 python speechToText_Muti.py GoogleSTT  634d6814ea4db11e1b3cb03d
    # python speechToText_Muti.py whisper 62befe7c27977f737525c4c3
    #  python speechToText_Muti.py whisper 634d6814ea4db11e1b3cb03d
#google no optioon/ default
# ssh lh3057@kryten.janakj.net python3  <  speechToText_Muti.py - GoogleSTT  634d6814ea4db11e1b3cb03d GOOGLE_STT_KEY=AIzaSyB-5VKtWsx7yCGCOxHRfpRDyZGjU8f4N80
#ssh user@machine python < script.py - arg1 arg2
    # cli( "GoogleSTT","62befe7c27977f737525c4c3")
    #python speechToText_Muti.py whisper 61fcaa1f2fc54df92426936e