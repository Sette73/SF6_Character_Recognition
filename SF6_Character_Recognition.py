from paddleocr import PaddleOCR
import paddle
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
import numpy as np
import cv2
from time import sleep
import time
import keras
import tensorflow as tf
import pygetwindow as gw
import pyautogui
import difflib
from selenium import webdriver
from selenium.webdriver.common.by import By
import pickle
import getpass
from obswebsocket import obsws, requests
import unicodedata

# ウィンドウタイトルを指定
window_title = "Street Fighter 6"

# 使用するメモリ量をMB単位で指定
MEMORY_LIMIT_MB = 8096  # 1GBを割り当てる場合

# GPUデバイスのリストを取得
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 指定されたサイズでメモリを静的に割り当てる
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=MEMORY_LIMIT_MB)]
            )
        print(f"Allocated {MEMORY_LIMIT_MB}MB of GPU memory.")
    except RuntimeError as e:
        print(e)

#分類モデルのロード
loaded_model = keras.models.load_model('screen_predict_model.h5')

#キャラクター名リスト (舞まで)
Character_Name = ["A.K.I.",      \
                  "BLANKA",      \
                  "CAMMY",       \
                  "CHUN-LI",     \
                  "DEE JAY",     \
                  "DHALSIM",     \
                  "ED",          \
                  "E.HONDA",     \
                  "GOUKI",       \
                  "GUILE",       \
                  "JAMIE",       \
                  "JP",          \
                  "JURI",        \
                  "KEN",         \
                  "KIMBERLY",    \
                  "LILY",        \
                  "LUKE",        \
                  "MAI",         \
                  "MANON",       \
                  "MARISA",      \
                  "RASHID",      \
                  "RYU",         \
                  "TERRY",       \
                  "VEGA",        \
                  "ZANGIEF"]

def Take_Screenshot(title):
    window = gw.getWindowsWithTitle(title)
    if not window:
        print("指定したウィンドウが見つかりません")
    else:
        # ウィンドウの位置とサイズを取得
        window = window[0]  # 最初の一致したウィンドウ
        left, top, width, height = window.left, window.top, window.width, window.height
        
        # ウィンドウ領域のスクリーンショットを取得
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        return screenshot
        # スクリーンショットを表示または保存
        #screenshot.save("window_screenshot.png")  # ファイルとして保存

def img_predict(image):
    img = image.convert("RGB")
    img = img.resize((1280,720))
    data = np.asarray(img)
    x = np.expand_dims(data, axis=0)
    return(loaded_model.predict(x))

#PaddleOCRを定義
ocr_en = PaddleOCR(
        use_gpu=False, #GPUあるならTrue
        lang = "en", #英語OCRならen
        det_limit_side_len=1920, #画像サイズが960に圧縮されないように必須設定
        )
ocr_japan = PaddleOCR(
        use_gpu=False, #GPUあるならTrue
        lang = "japan", #英語OCRならen
        det_limit_side_len=1920, #画像サイズが960に圧縮されないように必須設定
        )

def Image_Pretreatment(image,loc):
    Coordinates = [[90,805,550,895]    # P1キャラクター
                   ,[315,870,600,980]   # P1プレイヤーネーム
                   ,[1270,805,1830,895] # P2キャラクター 
                   ,[1340,870,1600,950]] # P2プレイヤーネーム
    im = image.crop(Coordinates[loc])
    im.convert('L')
    enhancer = ImageEnhance.Contrast(im)
    im_con = enhancer.enhance(2.0)
    np_img = np.array(im_con)
    
    return np_img

def run_ocr_matching(image):
    #画像読み込み＋前処理(適当)+PaddleOCR入力用にnpへ
    P1_Character_img = Image_Pretreatment(image,0)
    P1_Name_img = Image_Pretreatment(image,1)
    P2_Character_img = Image_Pretreatment(image,2)
    P2_Name_img = Image_Pretreatment(image,3)
    
    #PaddleOCRでOCR ※cls(傾き設定)は矩形全体での補正なので1文字1文字の補正ではない為不要
    P1_Character = ocr_en.ocr(img = P1_Character_img, det=True, rec=True, cls=False)
    P1_Name = ocr_japan.ocr(img = P1_Name_img, det=True, rec=True, cls=False)
    P2_Character = ocr_en.ocr(img = P2_Character_img, det=True, rec=True, cls=False)
    P2_Name = ocr_japan.ocr(img = P2_Name_img, det=True, rec=True, cls=False)

    str_list = ['','','','']
    if type(P1_Character[0]) == list:
        str_list[0] = P1_Character[0][0][1][0]
    if type(P1_Name[0]) == list:
        str_list[1] = unicodedata.normalize('NFKC',P1_Name[0][0][1][0])
    if type(P2_Character[0]) == list:
        str_list[2] = P2_Character[0][0][1][0]
    if type(P2_Name[0]) == list:
        str_list[3] = unicodedata.normalize('NFKC',P2_Name[0][0][1][0])

    return str_list

#selenium用設定
cookies_file = 'SF6.pkl'
usercode = "user-code"
eaddres = "e-mail@example.com"
options = webdriver.ChromeOptions()
useragent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
options.add_argument("--user-agent=" + useragent)
options.add_argument('--headless')
chrome = webdriver.Chrome(options=options)
cookies = pickle.load(open(cookies_file,'rb'))
chrome.implicitly_wait(2)


def cookies_verify(cookies):
    chrome.get("https://www.streetfighter.com/6/buckler/ja-jp/profile/" + usercode)
    for c in cookies:
        chrome.add_cookie(c)
    chrome.get("https://www.streetfighter.com/6/buckler/ja-jp/profile/" + usercode)

    try:
        chrome.find_element(By.ID,"CybotCookiebotDialogBodyButtonDecline").click()
        time.sleep(0.5)
    except Exception:
        pass


    if("<article class=\"not_registered" in chrome.page_source):
        chrome.find_element(By.CSS_SELECTOR,"article[class^=\"not_registered\"]")
        pswd = getpass.getpass(prompt="password: ")
        print(pswd)
        chrome.get("https://www.streetfighter.com/6/buckler/ja-jp/auth/loginep?redirect_url=/profile/auth")
        try:
            chrome.find_element(By.ID,"1-email").send_keys(eaddres)
            chrome.find_element(By.NAME,"password").send_keys(pswd)
            chrome.find_element(By.NAME,"submit").click()
            time.sleep(1)
        except Exception:
            chrome.save_screenshot('error.png')
            
        cookies = chrome.get_cookies()
        pickle.dump(cookies,open(cookies_file,'wb'))
        print('cookie used')
        
def Player_Search(name):
    chrome.get("https://www.streetfighter.com/6/buckler/ja-jp/fighterslist/search")
    chrome.get("https://www.streetfighter.com/6/buckler/ja-jp/fighterslist/search/result?fighter_id=" + name +"&page=1")
    try:
        a = chrome.find_element(By.CSS_SELECTOR,"ul[class^=list_fighter_list]")
        a.find_element(By.TAG_NAME,"a").click()
    except:
        return 0
    try:
        chrome.find_element(By.CSS_SELECTOR,"article[class^=\"status_title\"]")
        code = chrome.current_url.replace("https://www.streetfighter.com/6/buckler/ja-jp/profile/","")    
    except:
        return 0
            
    return code

def get_winrate(code):
    if code == 0:
        return {}
    chrome.get("https://www.streetfighter.com/6/buckler/ja-jp/profile/" + str(code) + "/play")
    try:
        chrome.find_element(By.CSS_SELECTOR,"aside[class^=\"play\"]").find_elements(By.TAG_NAME,"li")[4].click()
        chrome.find_element(By.CSS_SELECTOR,"aside[class^=\"filter_nav_filter_nav\"]").find_elements(By.TAG_NAME,"dd")[1].find_element(By.CSS_SELECTOR,"option[value=\"2\"]").click()
    except Exception:
        return {}
    f = chrome.find_element(By.CSS_SELECTOR,"article[class^=\"winning_rate_winning_rate\"]").find_elements(By.TAG_NAME,"p")
    
    dic = {}
    name = ''
    mcount = ''
    wrate = ''
    
    for i in range(4,len(f)):
        if i % 3 == 1:
            name = f[i].text
        elif i % 3 == 2:
            mcount = f[i].text
        elif i % 3 == 0:
            wrate = f[i].text
            dic[name] = [mcount,wrate]

    return dic
    

#obswebsocket用設定
host = "localhost"
port = 4455
password = "PASSWORD"
use_scene = 'SF6'

ws = obsws(host, port, password)
ws.connect()

def Set_1PWinrate(txt):
    ItemId = ws.call(requests.GetSceneItemId(sceneName = use_scene,sourceName = '1P_Winrate'))
    ws.call(requests.SetInputSettings(inputName='1P_Winrate',inputSettings={'text':txt}))
    ws.call(requests.SetSceneItemLocked(sceneName = use_scene,sceneItemId = ItemId.datain['sceneItemId'],sceneItemLocked = False))
    sleep(0.01)
    wid = ws.call(requests.GetSceneItemTransform(sceneName=use_scene,sceneItemId=ItemId.datain['sceneItemId'])).datain['sceneItemTransform']['width']
    x = (1920 - 1312) - wid
    ws.call(requests.SetSceneItemTransform(sceneName = use_scene,sceneItemId = ItemId.datain['sceneItemId'],sceneItemTransform={'positionX':x}))
    ws.call(requests.SetSceneItemLocked(sceneName = use_scene,sceneItemId = ItemId.datain['sceneItemId'],sceneItemLocked = True))


def Set_2PCharacter(name):
    txt = 'vs ' + name
    ws.call(requests.SetInputSettings(inputName='2P_Char',inputSettings={'text':txt}))

def Set_1PBtlcnt(txt):
    ItemId = ws.call(requests.GetSceneItemId(sceneName = use_scene,sourceName = '1P_Btlcnt'))
    ws.call(requests.SetInputSettings(inputName='1P_Btlcnt',inputSettings={'text':txt}))
    ws.call(requests.SetSceneItemLocked(sceneName = use_scene,sceneItemId = ItemId.datain['sceneItemId'],sceneItemLocked = False))
    sleep(0.01)
    wid = ws.call(requests.GetSceneItemTransform(sceneName=use_scene,sceneItemId=ItemId.datain['sceneItemId'])).datain['sceneItemTransform']['width']
    x = (1920 - 1170) - wid
    ws.call(requests.SetSceneItemTransform(sceneName = use_scene,sceneItemId = ItemId.datain['sceneItemId'],sceneItemTransform={'positionX':x}))
    ws.call(requests.SetSceneItemLocked(sceneName = use_scene,sceneItemId = ItemId.datain['sceneItemId'],sceneItemLocked = True))


def Set_2PWinrate(txt):
    #ItemId = ws.call(requests.GetSceneItemId(sceneName = use_scene,sourceName = '2P_Winrate'))
    ws.call(requests.SetInputSettings(inputName='2P_Winrate',inputSettings={'text':txt}))

def Set_1PCharacter(name):
    txt = 'vs ' + name
    ItemId = ws.call(requests.GetSceneItemId(sceneName = use_scene,sourceName = '1P_Char'))
    ws.call(requests.SetInputSettings(inputName='1P_Char',inputSettings={'text':txt}))
    ws.call(requests.SetSceneItemLocked(sceneName = use_scene,sceneItemId = ItemId.datain['sceneItemId'],sceneItemLocked = False))

    sleep(0.01)
    wid = ws.call(requests.GetSceneItemTransform(sceneName=use_scene,sceneItemId=ItemId.datain['sceneItemId'])).datain['sceneItemTransform']['width']
    x = (1920 - 252) - wid
    ws.call(requests.SetSceneItemTransform(sceneName = use_scene,sceneItemId = ItemId.datain['sceneItemId'],sceneItemTransform={'positionX':x}))
    ws.call(requests.SetSceneItemLocked(sceneName = use_scene,sceneItemId = ItemId.datain['sceneItemId'],sceneItemLocked = True))

def Set_2PBtlcnt(txt):
    #ItemId = ws.call(requests.GetSceneItemId(sceneName = use_scene,sourceName = '2P_Btlcnt'))
    ws.call(requests.SetInputSettings(inputName='2P_Btlcnt',inputSettings={'text':txt}))
    
def OBS_Source_Enabled(name):
    ItemId = ws.call(requests.GetSceneItemId(sceneName = use_scene,sourceName = name))
    ws.call(requests.SetSceneItemEnabled(sceneName = use_scene,sceneItemId = ItemId.datain['sceneItemId'],sceneItemEnabled=True))

def OBS_Source_Disabled(name):
    ItemId = ws.call(requests.GetSceneItemId(sceneName = use_scene,sourceName = name))
    ws.call(requests.SetSceneItemEnabled(sceneName = use_scene,sceneItemId = ItemId.datain['sceneItemId'],sceneItemEnabled=False))

def OBS_ALL_Enabled(data):
    OBS_Source_Enabled('固定')
    OBS_Source_Enabled('1P_Char')
    OBS_Source_Enabled('2P_Char')
    
    global p1p2
    global my_winratedict
    global enemy_winratedict
    
    if playerside == 0 and data[0] == False:
        Set_1PBtlcnt(my_winratedict[p1p2[1][1]][0])
        Set_1PWinrate(my_winratedict[p1p2[1][1]][1])
    if playerside == 0 and data[1] == False:
        Set_2PBtlcnt(enemy_winratedict[p1p2[0][1]][0])
        Set_2PWinrate(enemy_winratedict[p1p2[0][1]][1])
        print(p1p2[0][1],enemy_winratedict[p1p2[0][1]][1],enemy_winratedict[p1p2[0][1]][0])
    if playerside == 1 and data[0] == False:
        Set_1PBtlcnt(enemy_winratedict[p1p2[1][1]][0])
        Set_1PWinrate(enemy_winratedict[p1p2[1][1]][1])
        print(p1p2[1][1],enemy_winratedict[p1p2[1][1]][1],enemy_winratedict[p1p2[1][1]][0])
    if playerside == 1 and data[1] == False:
        Set_2PBtlcnt(my_winratedict[p1p2[0][1]][0])
        Set_2PWinrate(my_winratedict[p1p2[0][1]][1])
    if data[0] == False:
        OBS_Source_Enabled('1P_Winrate')
        OBS_Source_Enabled('1P_Btlcnt')
    elif data[0] == True:
        OBS_Source_Enabled('1PNODATA')
    if data[1] == False:
        OBS_Source_Enabled('2P_Winrate')
        OBS_Source_Enabled('2P_Btlcnt')
    elif data[1] == True:
        OBS_Source_Enabled('2PNODATA')

def OBS_ALL_Disabled():
    OBS_Source_Disabled('固定')
    OBS_Source_Disabled('1PNODATA')
    OBS_Source_Disabled('2PNODATA')
    OBS_Source_Disabled('1P_Char')
    OBS_Source_Disabled('2P_Char')
    OBS_Source_Disabled('1P_Winrate')
    OBS_Source_Disabled('2P_Winrate')
    OBS_Source_Disabled('1P_Btlcnt')
    OBS_Source_Disabled('2P_Btlcnt')


cookies_verify(cookies)
my_winratedict = get_winrate(usercode)
enemy_winratedict = {}

chrome.get("https://www.streetfighter.com/6/buckler/ja-jp/profile/" + usercode)
playername = chrome.find_element(By.CSS_SELECTOR,"span[class^=\"status_name\"]").text
playerside = 0
#相手の名前
enemycode = 1
#直前の画面状況
latest_screen = 0
#勝率取得済みかどうか
completed = False
#プレイヤー情報取得有無
Nodata = [False,False]
p1p2 = [[0,''],[0,'']]
OBS_ALL_Disabled()

while True:
    try:
        sleep(1)
        screenshot = Take_Screenshot(window_title)

        features = img_predict(screenshot) #categories = ["load","matching","start_fin","VS","battle","training","other"]
        print(features)
        if features[0][0] == 1: #load画面
            if latest_screen == 2 or latest_screen == 4:
                OBS_ALL_Disabled()
            print("now loading...")
            completed = False
        if features[0][1] == 1: #マッチング画面
            print("matching!")
        if features[0][2] == 1: #試合開始、終了
            my_winratedict = get_winrate(usercode)
            your_winratedict = get_winrate(enemycode)
            
        if features[0][3] == 1: #開始前画面
            if latest_screen == 3 and not completed:
                Nodata[0],Nodata[1] = False,False
                p1p2 = [[0,''],[0,'']]
                ls = run_ocr_matching(screenshot)
                if ls.count('') > 1:
                    continue
                if playername in ls:
                    playerside = 0 if ls.index(playername) < 2 else 1
                else:
                    sim = [0,0]
                    sim[0] = difflib.SequenceMatcher(None,playername,ls[1]).ratio()
                    sim[1] = difflib.SequenceMatcher(None,playername,ls[3]).ratio()
                    playerside = 0 if sim[0] > sim[1] else 1
                print(ls)
                for n in Character_Name:
                    Sim1 = difflib.SequenceMatcher(None,ls[0],n).ratio()
                    Sim2 = difflib.SequenceMatcher(None,ls[2],n).ratio()
                    if Sim1 > 0.5:
                        if Sim1 > p1p2[0][0]:
                            p1p2[0][0] = Sim1
                            p1p2[0][1] = n
                    if Sim2 > 0.5:
                        if Sim2 > p1p2[1][0]:
                            p1p2[1][0] = Sim2
                            p1p2[1][1] = n
                            
                Set_1PCharacter(p1p2[0][1])
                Set_2PCharacter(p1p2[1][1])
                
                if playerside == 0:
                    enemycode = Player_Search(ls[3])
                    enemy_winratedict = get_winrate(enemycode)
                    
                    Set_1PBtlcnt(my_winratedict[p1p2[1][1]][0])
                    Set_1PWinrate(my_winratedict[p1p2[1][1]][1])
                    
                    if enemy_winratedict != {}:
                        Set_2PBtlcnt(enemy_winratedict[p1p2[0][1]][0])
                        Set_2PWinrate(enemy_winratedict[p1p2[0][1]][1])
                    else:
                        Nodata[1] = True
                else:
                    enemycode = Player_Search(ls[1])
                    enemy_winratedict = get_winrate(enemycode)

                    Set_2PBtlcnt(my_winratedict[p1p2[0][1]][0])
                    Set_2PWinrate(my_winratedict[p1p2[0][1]][1])

                    if enemy_winratedict != {}:
                        Set_1PBtlcnt(enemy_winratedict[p1p2[1][1]][0])
                        Set_1PWinrate(enemy_winratedict[p1p2[1][1]][1])
                    else:
                        Nodata[0] = True
                completed = True
            
        if features[0][4] == 1: #試合中
            if latest_screen == 0 or latest_screen == 2 or latest_screen ==3 or latest_screen == 6:
                OBS_ALL_Enabled(Nodata)
            print("now fighting...")
        if features[0][5] == 1: #トレモ画面
            if latest_screen == 0 or features.tolist()[0].count(1) == 1:
                OBS_ALL_Disabled()
            print("now training...")
        if features[0][6] == 1: #その他画面
            print("other")
        try:
            latest_screen = features.tolist()[0].index(1)
        except Exception:
            latest_screen = features.tolist()[0].index(max(features.tolist()[0]))
            
    except KeyboardInterrupt:
        print('Fin.')
        break
    except Exception as e:
        print(e)
        continue
