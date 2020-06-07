import os
import io
import json

import responder

from color_design_format import (
    load_img, ColorDesignFormatter, save_img
)


env = os.environ
DEBUG = env['DEBUG'] in ['1', 'True', 'true']
METRIC = env['METRIC']
IMAGE_FORMAT = env['IMAGE_FORMAT']
MODE = env['MODE']

cur_dir = os.path.dirname(__file__)
with open(os.path.join(cur_dir, 'color_design_format.json')) as fp:
    CONFIG = json.load(fp)

api = responder.API(debug=DEBUG)
formatter = ColorDesignFormatter(
    color_design_format=CONFIG,
    metric=METRIC,
    mode=MODE
)


def format_color_design(bytes_io, mode='HSV'):
    img = load_img(bytes_io, mode=mode)
    formatted_img = formatter.format(img)
    return save_img(formatted_img, format_=IMAGE_FORMAT)


@api.route("/")
async def extract(req, resp):
    body = await req.content
    resp.content = format_color_design(
        io.BytesIO(body),
        mode=MODE
    )


if __name__ == "__main__":
    api.run()