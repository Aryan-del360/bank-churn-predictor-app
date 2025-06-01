# Bank Customer Churn Predictor

This project is all about helping banks figure out which customers might leave, so they can keep them happy! It's a full cycle: from crunching numbers to making a live app and even showing insights on a dashboard.

## What's Inside

* **Smart Predictions:** I built a machine learning model (XGBoost, specifically) that's really good at guessing if a customer will churn. It's about 88% accurate!
* **Live App:** There's a simple web app built with Streamlit where you can play around and see predictions in real-time.
* **Key Insights:** I also made an interactive dashboard in Power BI to show `why` customers might leave, giving banks clear actions to take.

### My Tools

* **Python:** My go-to language.
* **Libraries:** Used Pandas for data, Scikit-learn and XGBoost for the model, Streamlit for the app, and Matplotlib/Seaborn for some visuals.
* **Dashboards:** Power BI is what I used for the interactive insights.

## Project Files (What's in this folder)

```
.
├── data/                       # Raw customer info
├── notebooks/                  # My data exploration and model building steps
├── src/                        # Python code for data prep, training, and predicting
├── app/streamlit_app.py        # The fun web app
├── dashboards/                 # Power BI dashboard files
├── README.md                   # You're reading it!
├── requirements.txt            # All the Python libraries you need
├── .gitignore                  # Keeps my Git clean
```

## How to Get It Running (Locally)

Want to try it out? Here's how:

1.  **Grab the code:**
    ```bash
    git clone [https://github.com/Aryan-del360/bank-churn-predictor-app.git](https://github.com/Aryan-del360/bank-churn-predictor-app.git)
    cd bank-churn-predictor-app
    ```
2.  **Set up your Python space:**
    ```bash
    python -m venv venv
    # For Windows:
    .\venv\Scripts\activate
    # For Mac/Linux:
    source venv/bin/activate
    ```
3.  **Install everything needed:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Important NLTK download (if your app uses it):**
    ```python
    # Open Python
    import nltk
    nltk.download('vader_lexicon')
    # Type exit() to leave Python
    ```
5.  **Launch the web app!**
    ```bash
    streamlit run app/streamlit_app.py
    ```
    It should open in your browser, usually at `http://localhost:8501`.

## Using the Project

* **Train the Model:** Run `python src/train_model.py` to process data and train the prediction model.
* **See Predictions Live:** Open the Streamlit app (from step 5 above) and play around!
* **Explore Data:** Check out the `dashboards/churn_dashboard.pbix` file in Power BI to see the customer insights.

## How Well it Works

My XGBoost model hits about **88% accuracy** on predicting churn. You can find more detailed performance numbers (like precision, recall, etc.) in the `notebooks/churn_prediction_eda_modeling.ipynb` notebook.

## What's Next? (Future Ideas)

I'm always thinking about how to make this better:

* **Faster Predictions:** Build a quick API so other systems can get predictions instantly.
* **Smarter Features:** Dig deeper into the data or add new data sources to make predictions even stronger.
* **Explainable AI:** Add features to understand *why* the model predicts someone will churn, not just *if*.
* **Auto-Updates:** Set up a system to automatically retrain and deploy the model as new customer data comes in.

## Let's Connect!

Have questions or want to chat about data science? Reach out!

**Shubham Sharma**
* **Email:** shubhamdatascientist76@gmail.com
* **LinkedIn:** [My LinkedIn Profile](https://www.linkedin.com/in/shubham-sharma-224954367/)
* **GitHub:** [My GitHub Profile](https://github.com/Aryan-del360)
* **Portfolio:** [My Portfolio Website](https://your-portfolio-url.com) - *Remember to put your actual portfolio link here!*
```