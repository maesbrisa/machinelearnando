from flask import Flask, request
from flask_api import status

from predict import predict_phishing

app = Flask(__name__)


@app.route('/phising-ml/', methods=['POST'])
def check_phising():
    mail_content = request.form.get('mail')
    if mail_content is not None:
        results = predict_phishing(mail_content)[0].tolist()
        data = {'is_phising': results.index(max(results)) == 1,
                'probability': max(results)}
        return data, 200
    else:
        content = {'error': 'no mail provided'}
        return content, status.HTTP_400_BAD_REQUEST
