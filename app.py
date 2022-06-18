from flask import Flask, render_template, request, redirect
import pandas as pd
import librosa
import librosa.display
from antropy import *
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    disease = ""
    if request.method == "POST":
        disease==""
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            age = request.form['age']
            gender = request.form['gender']
            if gender == 'male':
                gender_encoded = 1
            else:
                gender_encoded = 0
            height = float(request.form['height']) # in meters
            weight = float(request.form['weight'])
            bmi = (weight / height ** 2) * 10000
            print(age, gender_encoded, weight, height)
            breathing_data_array = [age, gender_encoded, bmi]
            SAMPLE_RATE = 16000
            raw, sr = librosa.load(file, sr=SAMPLE_RATE, duration=20)
            mfccs = librosa.feature.mfcc(raw, hop_length=len(raw), n_mfcc=8)
            mfccs = mfccs.flatten()
            zcr = librosa.core.zero_crossings(raw).sum() / len(raw)
            sc = librosa.feature.spectral_centroid(raw)[0]
            rms = librosa.feature.rms(raw)[0]
            s_rf = librosa.feature.spectral_rolloff(raw, roll_percent=0.85)[0]
            s_rf_75 = librosa.feature.spectral_rolloff(raw, roll_percent=0.75)[0]
            sf = librosa.feature.spectral_flatness(raw)[0]
            se = entropy.spectral_entropy(x=raw, sf=sr, method='fft')

            breathing_data_array.extend([zcr, sc.mean(), np.median(sc), sc.std(), rms.mean(), np.median(rms),
                            rms.std(), s_rf.mean(), np.median(s_rf), s_rf.std(), s_rf_75.mean(),
                            np.median(s_rf_75), s_rf_75.std(), sf.mean(), np.median(sf), sf.std(), se])
            breathing_data_array.extend(mfccs)
            breathing_data_df = pd.DataFrame(breathing_data_array).T
            print(breathing_data_df.columns.tolist())

            print("form data received")

            with open('Decision_Tree_model.pkl', 'rb') as f:
                model = pickle.load(f)

            classes = pd.read_pickle('illness_classification.pkl')

            result_coded = model.predict(breathing_data_df)
            result = classes.loc[result_coded]
            disease=' '.join([str(elem) for elem in result.values.tolist()[-1]])



    return render_template('index.html', disease=disease)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
