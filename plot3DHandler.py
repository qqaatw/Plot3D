import logging

from plot3D.main import generate_plot
from flask import Flask, make_response, request, send_file, send_from_directory

app = Flask(__name__)
logging.getLogger().setLevel('INFO')
logging.getLogger('werkzeug').disabled = True

@app.route('/plot3D/<path:path>', methods=['GET'])
def plot3D(path):
    return send_from_directory('templates', path)

@app.route('/images/<path:path>', methods=['GET'])
def images(path):
    response = make_response(open("./plot3D/images/" + path, 'rb').read())
    response.content_type = "image"
    return response

@app.route('/get_testfile', methods=['GET'])
def get_testfile():
    return send_file('./plot3D/foo.png', mimetype='image/png')

@app.route('/handler', methods=['POST'])
def handler():
    dimention3D = []
    dimention2D = []
    advance3D = []
    for i in request.form:
        try:
            if request.form[i] == '':
                return make_response('nodata')
            else:
                p_type, p_sn = i.split('-')
                if p_type == 'A':
                    dimention3D.append(
                        {"A": request.form[i],
                         "B": request.form["B-"+p_sn],
                         "C": request.form["C-"+p_sn],
                         "Xexp": request.form["Xexp-"+p_sn],
                         "Yexp": request.form["Yexp-"+p_sn]
                        })
                if p_type == '2DA':
                    dimention2D.append(
                        {"A": request.form[i],
                         "C": request.form["2DC-"+p_sn],
                         "Z": request.form["2DZ-"+p_sn],
                         "Xexp": request.form["2DXexp-"+p_sn]})
                if p_type == 'F':
                    advance3D.append(request.form[i])
        except:
            return make_response('nodata')

    logging.info("Dimension2D Requests: {}".format(dimention2D))
    logging.info("Dimension3D Requests: {}".format(dimention3D))    
    logging.info("Advance3D Requests: {}".format(advance3D))

    uuid = generate_plot(dimention3D, dimention2D, advance3D)
    logging.info("UUID: {}".format(uuid))
    response = make_response(str(uuid))
    #response.headers['Access-Control-Allow-Origin'] = '*'
    return response


if __name__ == "__main__":
    app.run('0.0.0.0', 80)
