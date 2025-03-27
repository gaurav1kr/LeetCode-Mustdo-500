import os
import time
import pickle
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ✅ Path to Markdown File
MD_FILE_PATH = "/home/gakumar/LeetCode-Mustdo-500/Easy/final.md"

# ✅ Cookie storage file
COOKIE_FILE = "leetcode_cookies.pkl"

# ✅ Set up Microsoft Edge WebDriver
edge_options = Options()
edge_options.add_argument("--no-sandbox")  
edge_options.add_argument("--disable-dev-shm-usage")
edge_options.add_argument("--disable-gpu")  
edge_options.add_argument("--remote-debugging-port=9222")
edge_options.add_argument("--disable-software-rasterizer")
edge_options.add_argument("--user-data-dir=C:\\selenium\\edge-profile")
edge_options.add_argument("--profile-directory=Default")

print("🚀 Ensure you have started msedgedriver.exe on Windows with: msedgedriver.exe --port=9515")

try:
    driver = webdriver.Remote(
        command_executor="http://localhost:9515",
        options=edge_options
    )
except Exception as e:
    print(f"❌ Error initializing Edge WebDriver: {e}")
    exit(1)

# ✅ Function to save and load cookies
def save_cookies():
    pickle.dump(driver.get_cookies(), open(COOKIE_FILE, "wb"))
    print("✅ Cookies saved!")

def load_cookies():
    if os.path.exists(COOKIE_FILE):
        cookies = pickle.load(open(COOKIE_FILE, "rb"))
        for cookie in cookies:
            driver.add_cookie(cookie)
        print("✅ Cookies loaded!")

# ✅ Function to check if logged in
def is_logged_in():
    driver.get("https://leetcode.com/problemset/all/")
    time.sleep(5)

    load_cookies()
    driver.refresh()
    time.sleep(3)

    try:
        driver.find_element(By.XPATH, "//a[contains(@href, '/accounts/logout/')]")
        print("✅ Already logged in.")
        save_cookies()
        return True
    except:
        print("❌ Not logged in. Manual login required.")
        return False

# ✅ Function to parse Markdown file and extract problems
def parse_markdown(md_file):
    with open(md_file, "r", encoding="utf-8") as file:
        md_content = file.read()

    # 🔥 FIX: Handle "##1", "##2" (without space after ##)
    sections = re.split(r"\n##\d+", md_content)  

    problems = []

    for section in sections[1:]:  # Skip content before first problem
        lines = section.strip().split("\n")
        title = lines[0].strip() if lines else "UNKNOWN"

        # 🔥 FIX: Extract problem link correctly
        link_match = re.search(r"\*\*\*\*\[Problem Link\](https://leetcode\.com/problems/[\w\-]+)", section)
        link = link_match.group(1) if link_match else None

        # 🔥 FIX: Extract C++ code properly
        code = []
        is_code = False

        for line in lines:
            if line.startswith("```cpp"):
                is_code = True
                continue
            elif line.startswith("```"):  # End of code block
                is_code = False
                continue
            elif is_code:
                code.append(line)

        if link:
            problems.append({"title": title, "link": link, "code": "\n".join(code)})

    print(f"✅ Extracted {len(problems)} problems from Markdown.")
    return problems

# ✅ Function to submit solutions on LeetCode
def submit_solution(problem):
    print(f"🚀 Submitting: {problem['title']}")
    print(f"📌 Extracted Code for {problem['title']}:")
    print(problem['code'])

    driver.get(problem["link"])
    time.sleep(5)

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
            let monacoEditor = monaco.editor.getModels()[0];
            if (monacoEditor) {
                monacoEditor.setValue(arguments[0]);
            } else {
                let cmEditor = document.querySelector('.CodeMirror').CodeMirror;
                cmEditor.setValue(arguments[0]);
            }
        """, problem["code"])

        time.sleep(5)  # Ensure code is fully inserted
        
        # ✅ Simulate Ctrl+Enter to submit the solution
        driver.execute_script("""
            let event = new KeyboardEvent('keydown', {
                key: 'Enter',
                code: 'Enter',
                keyCode: 13,
                which: 13,
                ctrlKey: true,
                bubbles: true
            });
            document.activeElement.dispatchEvent(event);
        """)
        print("🚀 Simulated Ctrl+Enter using JavaScript!")

        time.sleep(10)  # Wait for submission
    except Exception as e:
        print(f"❌ Error submitting solution: {e}")
        driver.save_screenshot("submit_debug.png")
        print("📸 Screenshot saved as 'submit_debug.png' - Check if Submit button exists!")

# ✅ Main execution
if __name__ == "__main__":
    if not is_logged_in():
        print("⚠️ Please log in manually in the opened browser.")
        input("🔵 Press Enter after logging in...")
    
    problems = parse_markdown(MD_FILE_PATH)
    
    for problem in problems:
        submit_solution(problem)

    print("🎉 All problems submitted!")
    driver.quit()

