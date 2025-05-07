import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Setup Chrome driver
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Form URL
form_url = "https://forms.office.com/Pages/ResponsePage.aspx?id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAANAAUvxT25UQ1cwVTdZUjc5UDBKMFVVNTE5OVJXSENKOS4u"

# Options for random selection
education_options = ['Graduate', 'High School', 'Postgrad']
family_type_options = ['Nuclear', 'Joint']
siblings_options = ['0', '1', '2', '3', 'more than 3']
guardian_options = ['Parents', 'Single Parent', 'Others']
study_env_options = ['Noisy', 'Quiet', 'Shared Room']
residential_options = ['Hostel', 'Home', 'PG']
domestic_issues_options = ['True', 'False']
stress_level_options = ['Stress', 'Normal']

def fill_form():
    driver.get(form_url)

    # Wait for page to load
    time.sleep(2)

    # 1. Parental income (text field)
    income_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//input[@type="text"]'))
    )
    income_value = str(random.randint(10000, 100000))
    income_input.send_keys(income_value)

    # 2. Parents education level (choice)
    edu_choice = random.choice(education_options)
    driver.find_element(By.XPATH, f'//div[contains(text(),"{edu_choice}")]').click()

    # 3. Family type
    family_choice = random.choice(family_type_options)
    driver.find_element(By.XPATH, f'//div[contains(text(),"{family_choice}")]').click()

    # 4. No. of siblings
    siblings_choice = random.choice(siblings_options)
    driver.find_element(By.XPATH, f'//div[contains(text(),"{siblings_choice}")]').click()

    # 5. Guardian type
    guardian_choice = random.choice(guardian_options)
    driver.find_element(By.XPATH, f'//div[contains(text(),"{guardian_choice}")]').click()

    # 6. Home study environment
    study_env_choice = random.choice(study_env_options)
    driver.find_element(By.XPATH, f'//div[contains(text(),"{study_env_choice}")]').click()

    # 7. Residential status
    residential_choice = random.choice(residential_options)
    driver.find_element(By.XPATH, f'//div[contains(text(),"{residential_choice}")]').click()

    # 8. Domestic issues reported
    domestic_choice = random.choice(domestic_issues_options)
    driver.find_element(By.XPATH, f'//div[contains(text(),"{domestic_choice}")]').click()

    # 9. Stress level
    stress_choice = random.choice(stress_level_options)
    driver.find_element(By.XPATH, f'//div[contains(text(),"{stress_choice}")]').click()

    # Submit the form
    submit_btn = driver.find_element(By.XPATH, '//button[@type="submit"]')
    submit_btn.click()

    # Wait for confirmation page
    time.sleep(2)

for i in range(200):
    print(f"Submitting form {i+1}")
    fill_form()
    time.sleep(2)  # Small wait before next submission
