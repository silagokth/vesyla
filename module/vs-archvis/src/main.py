import svg
import logging
import json
import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import argparse

def create_controller_library():
    library = {
        "geometry":{
            "width": 80,
            "height": 320,
        },
        "elements": {}
    }
    return library

def create_resource_library():
    library = {
        "geometry":{
            "width": 260,
            "height": 20,
        },
        "elements": {}
    }
    return library

def create_cell_library():
    library = {
        "geometry":{
            "width": 400,
            "height": 400,
        },
        "elements": {}
    }
    return library

def add_components(dir, controller_library, resource_library, cell_library):
    # scan recursively the directory, and find all files named as "arch.json"
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file == "arch.json":
                # read the json file
                with open(os.path.join(root, file), "r") as f:
                    arch = json.load(f)
                
                # check if arch is a json object
                if not isinstance(arch, dict):
                    logging.error("The architecture file is not a json object")
                    exit(1)
                
                # check if it has a name and a type
                if "name" not in arch:
                    logging.error("The architecture file does not have a name")
                    exit(1)
                if "type" not in arch:
                    logging.error("The architecture file does not have a type")
                    exit(1)
                
                # check if the type is controller or resource
                if arch["type"] == "controller":
                    # add the component to the library
                    controller_library["elements"][arch["name"]] = arch
                elif arch["type"] == "resource":
                    # add the component to the library
                    resource_library["elements"][arch["name"]] = arch
                elif arch["type"] == "cell":
                    # add the component to the library
                    cell_library["elements"][arch["name"]] = arch
                else:
                    logging.error("The architecture file has an invalid type: {0}".format(arch["type"]))
                    exit(1)


def read_architecture(file):
    # load the json file
    with open(file, "r") as f:
        j = json.load(f)
        # convert it to a dictionary
        return j

def add_custom_components(j, controller_library, resource_library, cell_library):
    if j.get("resources") is not None:
        for resource in j["resources"]:
            resource_name = resource["name"]
            resource_library["elements"][resource_name] = resource
    if j.get("controllers") is not None:
        for controller in j["controllers"]:
            controller_name = controller["name"]
            controller_library["elements"][controller_name] = controller
    if j.get("cells") is not None:
        for cell in j["cells"]:
            cell_library["elements"][cell["name"]] = cell

            # check if the components are in the library
            if cell["controller"] not in controller_library["elements"]:
                logging.error("Controller {0} is not in the library".format(cell["controller"]))
                exit(1)
            for resource in cell["resource_list"]:
                if resource not in resource_library["elements"]:
                    logging.error("Resource {0} is not in the library".format(resource))
                    exit(1)




def draw_resource(svg_obj, x, y, resource_type, controller_library, resource_library, cell_library):
    resource_properties = resource_library["elements"][resource_type]
    resource_width = resource_library["geometry"]["width"]
    resource_height = resource_library["geometry"]["height"]
    resource_size = resource_properties["size"]
    resource_color_fill = "#C7EBBA" if resource_properties.get("visual") is None or resource_properties["visual"].get("color_fill") is None else resource_properties["visual"]["color_fill"]
    resource_color_border = "#0D4A21" if resource_properties.get("visual") is None or resource_properties["visual"].get("color_border") is None else resource_properties["visual"]["color_border"]
    resource_color_text = "#0D4A21" if resource_properties.get("visual") is None or resource_properties["visual"].get("color_text") is None else resource_properties["visual"]["color_text"]

    svg_obj["elements"].append(svg.Rect(x=x+11, y=y+1, width=resource_width-12, height=resource_height*resource_size-2, fill=resource_color_fill, stroke=resource_color_border, stroke_width=2, rx=5, ry=5))
    text_x = x+11+resource_width/2
    text_y = y+5+resource_height*resource_size/2
    svg_obj["elements"].append(svg.Text(x=text_x, y=text_y, text=resource_type, fill=resource_color_text, font_size=16, text_anchor="middle"))

def draw_controller(svg_obj, x, y, controller_type, controller_library, resource_library, cell_library):
    controller_properties = controller_library["elements"][controller_type]
    controller_width = controller_library["geometry"]["width"]
    controller_height = controller_library["geometry"]["height"]
    controller_color_fill = "#FFF0B0" if controller_properties.get("visual") is None or controller_properties["visual"].get("color_fill") is None else controller_properties["visual"]["color_fill"]
    controller_color_border = "#A65900" if controller_properties.get("visual") is None or controller_properties["visual"].get("color_border") is None else controller_properties["visual"]["color_border"]
    controller_color_text = "#A65900" if controller_properties.get("visual") is None or controller_properties["visual"].get("color_text") is None else controller_properties["visual"]["color_text"]

    svg_obj["elements"].append(svg.Rect(x=x+11, y=y+11, width=controller_width-2, height=controller_height-2, fill=controller_color_fill, stroke=controller_color_border, stroke_width=2, rx=10, ry=10))
    text_x = x+11+controller_width/2
    text_y = y+11+controller_height/2
    # write text and rotate it 90 degrees
    svg_obj["elements"].append(svg.Text(x=text_x, y=text_y, text=controller_type, fill=controller_color_text, font_size=20, text_anchor="middle", transform="rotate(90, {0}, {1})".format(text_x, text_y)))

def draw_cell(svg_obj, x, y, cell_type, coord, controller_library, resource_library, cell_library):
    offset = 5
    cell_width = cell_library["geometry"]["width"]
    cell_height = cell_library["geometry"]["height"]
    cell_color_fill = "#E6E6E6"
    cell_color_border = "#A5A5A5"
    cell_color_text = "#323232"

    if cell_type == "":
        svg_obj["elements"].append(svg.Rect(x=x+offset, y=y+offset, width=cell_width-2*offset, height=cell_height-2*offset, fill=cell_color_fill, stroke=cell_color_border, stroke_width=2))
        # draw text to show coordinates
        xx = x + cell_width/2
        yy = y + 40
        svg_obj["elements"].append(svg.Text(x=xx, y=yy, text="[{0}, {1}]".format(coord[0], coord[1]), fill=cell_color_text, font_size=20, font_weight="bold", text_anchor="middle", font_style="normal"))
    else:
        cell_properties = cell_library["elements"][cell_type]
        svg_obj["elements"].append(svg.Rect(x=x+offset, y=y+offset, width=cell_width-2*offset, height=cell_height-2*offset, fill=cell_color_fill, stroke=cell_color_border, stroke_width=2))
        # draw text to show coordinates
        xx = x + cell_width/2
        yy = y + 40
        svg_obj["elements"].append(svg.Text(x=xx, y=yy, text="[{0}, {1}]".format(coord[0], coord[1]), fill=cell_color_text, font_size=20, font_weight="bold", text_anchor="middle", font_style="normal"))
        
        xx = x + offset + 10
        yy = y + offset + 50
        draw_controller(svg_obj, xx, yy, cell_properties["controller"], controller_library, resource_library, cell_library)
        curr_yy = yy = y + offset + 60
        for i, resource in enumerate(cell_properties["resource_list"]):
            xx = x + 100
            draw_resource(svg_obj, xx, yy, resource, controller_library, resource_library, cell_library)
            resource_properties = resource_library["elements"][resource]
            resource_height = resource_library["geometry"]["height"]
            resource_size = resource_properties["size"]
            yy += resource_height*resource_size

def draw_cells(svg_obj, row, col, cell_list, controller_library, resource_library, cell_library):
    coord_set = set()
    for r in range(row):
        for c in range(col):
            coord_set.add((r, c))
    for cell_type in cell_list:
        for i, cell in enumerate(cell_list[cell_type]):
            x = cell[1]*cell_library["geometry"]["width"]
            y = cell[0]*cell_library["geometry"]["height"] + 100
            coord = (cell[0], cell[1])
            draw_cell(svg_obj, x, y, cell_type, coord, controller_library, resource_library, cell_library)
            coord_set.remove(coord)
    for coord in coord_set:
        x = coord[1]*cell_library["geometry"]["width"]
        y = coord[0]*cell_library["geometry"]["height"] + 100
        draw_cell(svg_obj, x, y, "", coord, controller_library, resource_library, cell_library)

def draw_io_buffers(svg_obj, row, col, input_buffer_depth, output_buffer_depth, controller_library, resource_library, cell_library):
    offset = 5
    buffer_width = col * cell_library["geometry"]["width"]
    buffer_height = 100
    buffer_color_fill = "#B2E0E0"
    buffer_color_border = "#1C434C"
    buffer_color_text = "#1C434C"
    # input buffer
    svg_obj["elements"].append(svg.Rect(x=offset, y=offset, width=buffer_width-2*offset, height=buffer_height-2*offset, fill=buffer_color_fill, stroke=buffer_color_border, stroke_width=2))
    text_x = buffer_width/2
    text_y = buffer_height/2
    svg_obj["elements"].append(svg.Text(x=text_x, y=text_y, text="Input Buffer ({}x256b)".format(input_buffer_depth), fill=buffer_color_text, font_size=20, text_anchor="middle"))
    # output buffer
    yy = buffer_height + row * cell_library["geometry"]["height"]
    svg_obj["elements"].append(svg.Rect(x=offset, y=offset+yy, width=buffer_width-2*offset, height=buffer_height-2*offset, fill=buffer_color_fill, stroke=buffer_color_border, stroke_width=2))
    text_x = buffer_width/2
    text_y = yy + buffer_height/2
    svg_obj["elements"].append(svg.Text(x=text_x, y=text_y, text="Output Buffer ({}x256b)".format(output_buffer_depth), fill=buffer_color_text, font_size=20, text_anchor="middle"))

def draw(arch, output_dir, controller_library, resource_library, cell_library):
    cell_list = {}

    for cl in arch["fabric"]["cell_lists"]:
        cell_name = cl["cell_name"]
        coordinates = cl["coordinates"]
        coord = [(c["row"], c["col"]) for c in coordinates]
        if cell_name not in cell_list:
            cell_list[cell_name] = []
        cell_list[cell_name].extend(coord)

    svg_obj = {}
    svg_obj["width"] = cell_library["geometry"]["width"] * arch["fabric"]["width"]
    svg_obj["height"] = cell_library["geometry"]["height"] * arch["fabric"]["height"] + 200
    svg_obj["elements"] = []

    draw_io_buffers(svg_obj, arch["fabric"]["height"], arch["fabric"]["width"], arch["interface"]["input_buffer_depth"], arch["interface"]["output_buffer_depth"], controller_library, resource_library, cell_library)
    draw_cells(svg_obj, arch["fabric"]["height"], arch["fabric"]["width"], cell_list, controller_library, resource_library, cell_library)

    # create a canvas from string
    canvas = svg.SVG(width=svg_obj["width"], height=svg_obj["height"], elements=svg_obj["elements"])

    # save canvas to a file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "arch.svg"), "w+") as f:
        print(canvas, file=f)

    drawing = svg2rlg(os.path.join(output_dir, "arch.svg"))
    renderPDF.drawToFile(drawing, os.path.join(output_dir, "arch.pdf"))

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Visualize the architecture")
    parser.add_argument("-i", "--input", type=str, help="Path to the architecture file", default="arch.json")
    parser.add_argument("-o", "--output", type=str, help="Path to the output directory", default=".")
    parser.add_argument("-c", "--components", type=str, help="Path to the components directory", default=None)
    args = parser.parse_args()

    controller_library = create_controller_library()
    resource_library = create_resource_library()
    cell_library = create_cell_library()
    arch = read_architecture(args.input)
    if args.components is not None:
        add_components(args.components, controller_library, resource_library, cell_library)
    add_custom_components(arch, controller_library, resource_library, cell_library)
    draw(arch, args.output, controller_library, resource_library, cell_library)

if __name__ == "__main__":
    main()