from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait


class BookSpiritPage:
    QUESTION = (By.CSS_SELECTOR, "[data-testid='question-input']")
    MODE = (By.CSS_SELECTOR, "[data-testid='mode-select']")
    STRATEGY = (By.CSS_SELECTOR, "[data-testid='strategy-select']")
    SUBMIT = (By.CSS_SELECTOR, "[data-testid='submit-button']")
    STATUS = (By.CSS_SELECTOR, "[data-testid='status']")
    ANSWER = (By.CSS_SELECTOR, "[data-testid='answer']")
    SOURCE_CARDS = (By.CSS_SELECTOR, "[data-testid='source-card']")

    def __init__(self, driver, base_url: str, timeout: float = 8):
        self.driver = driver
        self.base_url = base_url.rstrip("/")
        self.wait = WebDriverWait(driver, timeout)

    def open(self):
        self.driver.get(f"{self.base_url}/demo")
        self.wait.until(EC.title_contains("Book Spirit"))
        self.wait.until(EC.visibility_of_element_located(self.QUESTION))
        return self

    def search(self, question: str, strategy: str = "vector"):
        question_input = self.driver.find_element(*self.QUESTION)
        question_input.clear()
        question_input.send_keys(question)
        Select(self.driver.find_element(*self.MODE)).select_by_value("search")
        Select(self.driver.find_element(*self.STRATEGY)).select_by_value(strategy)
        self.driver.find_element(*self.SUBMIT).click()
        self.wait.until(lambda driver: "sources" in driver.find_element(*self.STATUS).text)
        return self

    def answer_text(self) -> str:
        return self.driver.find_element(*self.ANSWER).text

    def source_texts(self) -> list[str]:
        return [element.text for element in self.driver.find_elements(*self.SOURCE_CARDS)]
