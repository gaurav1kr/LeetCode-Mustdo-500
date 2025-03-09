import os
import time
import pickle
import markdown2
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Load Markdown File
MD_FILE_PATH = "final.md"  # Update if needed

# Cookie storage file
COOKIE_FILE = "leetcode_cookies.pkl"

# Set up Microsoft Edge WebDriver to connect to remote instance on Windows
edge_options = Options()
edge_options.add_argument("--no-sandbox")  # Fixes crashes in WSL
edge_options.add_argument("--disable-dev-shm-usage")  # Prevents memory issues
edge_options.add_argument("--disable-gpu")  # Needed for some WSL setups
edge_options.add_argument("--remote-debugging-port=9222")  # Useful for debugging
edge_options.add_argument("--disable-software-rasterizer")
edge_options.add_argument("--user-data-dir=C:\\selenium\\edge-profile")  # Use pre-logged-in session
edge_options.add_argument("--profile-directory=Default")

# Ensure WebDriver is running on Windows before connecting
print("🚀 Ensure you have started msedgedriver.exe on Windows with: msedgedriver.exe --port=9515")

try:
    driver = webdriver.Remote(
        command_executor="http://localhost:9515",
        options=edge_options
    )
except Exception as e:
    print(f"❌ Error initializing Edge WebDriver: {e}")
    exit(1)

# Function to save and load cookies
def save_cookies():
    pickle.dump(driver.get_cookies(), open(COOKIE_FILE, "wb"))
    print("✅ Cookies saved!")

def load_cookies():
    if os.path.exists(COOKIE_FILE):
        cookies = pickle.load(open(COOKIE_FILE, "rb"))
        for cookie in cookies:
            driver.add_cookie(cookie)
        print("✅ Cookies loaded!")

# Function to check if logged in
def is_logged_in():
    driver.get("https://leetcode.com/problemset/all/")
    time.sleep(5)

    load_cookies()  # Load stored cookies before checking login
    driver.refresh()
    time.sleep(3)

    try:
        driver.find_element(By.XPATH, "//a[contains(@href, '/accounts/logout/')]")
        print("✅ Already logged in.")
        save_cookies()  # Save cookies after successful login
        return True
    except:
        print("❌ Not logged in. Manual login required.")
        return False

# Function to parse Markdown file and extract problems
def parse_markdown(md_file):
    with open(md_file, "r", encoding="utf-8") as file:
        md_content = file.read()

    problems = []
    sections = md_content.split("\n## ")  # Splitting by problem sections
    
    for section in sections[1:]:  # Ignore first split since it would be header
        lines = section.split("\n")
        title = lines[0].strip()
        link = None
        code = []
        is_code = False
        
        for line in lines[1:]:
            if line.startswith("**[Problem Link]"):
                link = line.split("(")[1].split(")")[0]
            elif line.startswith("```cpp"):
                is_code = True
            elif line.startswith("```"):
                is_code = False
            elif is_code:
                code.append(line)
        
        if link and code:
            problems.append({"title": title, "link": link, "code": "\n".join(code)})

    print(f"✅ Extracted {len(problems)} problems from Markdown.")
    return problems

# Function to submit solutions on LeetCode
def submit_solution(problem):
    print(f"🚀 Submitting: {problem['title']}")
    print(f"📌 Extracted Code for {problem['title']}:")
    print(problem['code'])

    driver.get(problem["link"])
    time.sleep(5)  # Wait for page to load

    driver.save_screenshot("editor_debug.png")
    print("📸 Screenshot saved as 'editor_debug.png' - Check if the editor is visible!")

    try:
        # Ensure editor is fully loaded
        editor_container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".monaco-editor, .CodeMirror"))
        )
        driver.execute_script("arguments[0].scrollIntoView();", editor_container)
        time.sleep(3)
        
        # Clear existing code and insert new solution
        driver.execute_script("""
            let editor = document.querySelector('.monaco-editor') || document.querySelector('.CodeMirror');
            if (editor.CodeMirror) {
                editor.CodeMirror.setValue('');
                editor.CodeMirror.setValue(arguments[0]);
            } else {
                let monacoEditor = monaco.editor.getModels()[0];
                monacoEditor.setValue('');
                monacoEditor.setValue(arguments[0]);
            }
        """, problem["code"])

        time.sleep(3)  # Ensure code is fully inserted
        
        # Verify inserted code
        extracted_code = driver.execute_script("""
            let editor = document.querySelector('.monaco-editor') || document.querySelector('.CodeMirror');
            if (editor.CodeMirror) {
                return editor.CodeMirror.getValue();
            } else {
                let monacoEditor = monaco.editor.getModels()[0];
                return monacoEditor.getValue();
            }
        """)
        print(f"🔍 Code inside editor after insertion:\n{extracted_code}")
        
        # Click Submit button
        submit_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//button[contains(text(),'Submit')]")
        ))
        driver.execute_script("arguments[0].scrollIntoView();", submit_button)
        time.sleep(2)
        submit_button.click()
        print("✅ Submit button clicked!")
        time.sleep(10)  # Wait for submission
    except Exception as e:
        print(f"❌ Error clicking Submit button: {e}")
        driver.save_screenshot("submit_debug.png")
        print("📸 Screenshot saved as 'submit_debug.png' - Check if Submit button exists!")

# Main execution
if __name__ == "__main__":
    if not is_logged_in():
        print("⚠️ Please log in manually in the opened browser.")
        input("🔵 Press Enter after logging in...")
    
    problems = parse_markdown(MD_FILE_PATH)
    
    for problem in problems:
        submit_solution(problem)

    print("🎉 All problems submitted!")
    driver.quit()

## On windows powershell , run below command on windows power shell -   & "C:\Program Files (x86)\Microsoft\Edge\Application\msedgedriver.exe" --port=9515 
