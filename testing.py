from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# resume = open('resume.pdf', 'r')
driver = webdriver.Firefox()
driver.get("http://127.0.0.1:5000/candidate_button")
driver.find_element(By.ID, "uploadBtn").click()
driver.get("http://127.0.0.1:5000/resume_form")
WebDriverWait(driver,3).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div/form/div[1]/button'))).click()
for i in range(2,10,1):
    WebDriverWait(driver,3).until(EC.element_to_be_clickable((By.XPATH, f'/html/body/div/form/div[{i}]/button[2]'))).click()
driver.find_element(By.ID, "sub").click()
driver.close()
