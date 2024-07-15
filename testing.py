from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
driver = webdriver.Firefox()
_200 = 0
_400 = 0
log = []
itr = 2 # change according to number of uploads
for i in range(itr):
    driver.get("http://127.0.0.1:5000/candidate_button")
    driver.find_element(By.ID, "uploadBtn").click()
    driver.get("http://127.0.0.1:5000/resume_form")
    WebDriverWait(driver,3).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div/form/div[1]/button'))).click()
    for j in range(2,10,1):
        WebDriverWait(driver,3).until(EC.element_to_be_clickable((By.XPATH, f'/html/body/div/form/div[{j}]/button[2]'))).click()
    driver.find_element(By.ID, "sub").click()
    response = requests.get('http://127.0.0.1:5000/resume_form')
    if response.status_code == 200:
        _200+=1
    elif response.status_code == 400:
        _400+=1
    else:
        log.append(response.status_code)

driver.close()
print(f'In {itr} iterations -')
print('Number of 200 responses = ',_200)
print('Number of 400 responses = ',_400)
print('Number of other responses = ', len(log))
print('Other responses = ',log)


# USE THE QUERIES BELOW TO CLEAN TABLES.
'''
SET FOREIGN_KEY_CHECKS = 0;
truncate table achievements;
truncate table certification_details;
truncate table education_details;
truncate table extracurricular_activities;
truncate table language_competencies;
truncate table personal_information;
truncate table project_details;
truncate table skills;
truncate table summary;
truncate table work_experience;
SET FOREIGN_KEY_CHECKS = 1;
'''