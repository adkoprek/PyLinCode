import sys
import os
import shutil
import selenium.webdriver as sw
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pyperclip
import time


MENU_XPATH = "/html/body/div[1]/div[1]/div/div[2]/div[1]/nav/div[2]/button/div"
EXPORT_IPORT_XPATH = "/html/body/div[1]/div[1]/div[1]/div[3]/div/div[2]/div[1]/a[9]/div[2]"
EXPORT_XPATH = "/html/body/div[1]/div[1]/div[1]/div[3]/div/div[2]/div[1]/a[2]"
EXPORT_BUTTON_XPATH = "/html/body/div[1]/div[2]/div[2]/div/div[2]/button[3]"
DOWNLOAD_DIRECTORY = "Downloads"

def start_session() -> sw.Firefox:
    driver = sw.Firefox()
    driver.get("https://stackedit.io/app")
    time.sleep(5)
    return driver

def insert_text(driver: sw.Firefox, text: str):
    pyperclip.copy(text)
    editor = driver.switch_to.active_element
    editor.send_keys(Keys.CONTROL, "a")
    editor.send_keys(Keys.DELETE)
    editor.send_keys(Keys.CONTROL, "v")

def download(driver: sw.Firefox):
    menu = driver.find_element(By.XPATH, MENU_XPATH)
    menu.click()
    export_import = driver.find_element(By.XPATH, EXPORT_IPORT_XPATH)
    export_import.click()
    export = driver.find_element(By.XPATH, EXPORT_XPATH)
    export.click()
    export_button = driver.find_element(By.XPATH, EXPORT_BUTTON_XPATH)
    export_button.click()
    time.sleep(1)

def move_file(input_file: str) -> str:
    path = os.path.join(os.path.expanduser("~"), DOWNLOAD_DIRECTORY)
    files = [os.path.join(path, f) for f in os.listdir(path)]
    if not files:
        return None
    file = max(files, key=os.path.getctime)
    
    dest = __file__.replace("converter.py", input_file.replace(".md", ".html"))
    shutil.move(file, dest)

    return dest

def strip(file_path: str):
    with open(file_path, 'r') as file:
        content = file.read()

    soup = BeautifulSoup(content, "html.parser")
    body = soup.find("body").find_all(recursive=False)[0]

    with open(file_path, 'w') as file:
        file.write(str(body))

if __name__ == '__main__':
    file_path = sys.argv[1]
    with open(file_path, 'r') as file:
        content = file.read()

    driver = start_session()
    insert_text(driver, content)
    download(driver)
    destination = move_file(file_path)
    strip(destination)
    driver.close()
