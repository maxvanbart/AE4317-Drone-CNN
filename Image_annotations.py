import labelbox as lb
import labelbox.data.annotation_types as lb_types
import uuid
import numpy as np

API_KEY = ""
client = lb.Client(API_KEY)

#IMPORT DATA ROWS

# send a sample image as batch to the project
global_key="2560px-Kitano_Street_Kobe01s5s41102.jpeg"

test_img_url = {
    "row_data": "https://storage.googleapis.com/labelbox-datasets/image_sample_data/2560px-Kitano_Street_Kobe01s5s4110.jpeg" ,
    "global_key": global_key
}


dataset = client.create_dataset(name="MAV_objects")
task = dataset.create_data_rows([test_img_url])
task.wait_till_done()
print("Errors:",task.errors)
print("Failed data rows:", task.failed_data_rows)


#BUILD ONTOLOGY

ontology_builder = lb.OntologyBuilder(
  classifications=[ # List of Classification objects
    lb.Classification(
      class_type=lb.Classification.Type.RADIO,
      name="pillar",
      options=[
        lb.Option(value="first_radio_answer"),
        lb.Option(value="second_radio_answer")
      ]
    ),
    lb.Classification(
      class_type=lb.Classification.Type.CHECKLIST,
      name="plant",
      options=[
        lb.Option(value="first_checklist_answer"),
        lb.Option(value="second_checklist_answer")
      ]
    ),
    lb.Classification(
      class_type=lb.Classification.Type.TEXT,
      name="QRcode"
    ),
    lb.Classification(
        class_type=lb.Classification.Type.RADIO,
        name="black board",
        options=[
            lb.Option("first_radio_answer",
                options=[
                    lb.Classification(
                        class_type=lb.Classification.Type.RADIO,
                        name="sub_radio_question",
                        options=[lb.Option("first_sub_radio_answer")]
                    )
                ]
            )
          ]
        ),
    lb.Classification(
      class_type=lb.Classification.Type.CHECKLIST,
      name="white board",
      options=[
          lb.Option("first_checklist_answer",
            options=[
              lb.Classification(
                  class_type=lb.Classification.Type.CHECKLIST,
                  name="sub_checklist_question",
                  options=[lb.Option("first_sub_checklist_answer")]
              )
          ]
        )
      ]
    ),
    lb.Classification(
          class_type=lb.Classification.Type.CHECKLIST,
          name="white board",
          options=[
              lb.Option("first_checklist_answer",
                options=[
                  lb.Classification(
                      class_type=lb.Classification.Type.CHECKLIST,
                      name="sub_checklist_question",
                      options=[lb.Option("first_sub_checklist_answer")]
                  )
              ]
            )
          ]
        ),
  ],
  tools=[ # List of Tool objects
    lb.Tool(
      tool=lb.Tool.Type.BBOX,
      name="bounding_box"),
    lb.Tool(
      tool=lb.Tool.Type.BBOX,
      name="bbox_with_radio_subclass",
      classifications=[
            lb.Classification(
                class_type=lb.Classification.Type.RADIO,
                name="sub_radio_question",
                options=[
                  lb.Option(value="first_sub_radio_answer")
                ]
              ),
        ]
      ),
    lb.Tool(
      tool=lb.Tool.Type.POLYGON,
      name="polygon"),
    lb.Tool(
      tool=lb.Tool.Type.SEGMENTATION,
      name="mask"),
      lb.Tool(
      tool=lb.Tool.Type.POINT,
      name="point"),
    lb.Tool(
      tool=lb.Tool.Type.LINE,
      name="polyline"),
    lb.Tool(
      tool=lb.Tool.Type.RELATIONSHIP,
      name="relationship")]
)

ontology = client.create_ontology("Image Prediction Import Demo", ontology_builder.asdict(), media_type=lb.MediaType.Image)