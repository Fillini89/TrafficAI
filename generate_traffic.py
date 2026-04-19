import xml.etree.ElementTree as ET
import xml.dom.minidom
from config import TRAFFIC_PHASES

def generate_dynamic_routes(filename="routes.rou.xml"):
    routes = ET.Element("routes")
    routes.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    routes.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")

    edges = [
        {"id": "f_0", "from": "-31272#6", "to": "-31272#7"}, {"id": "f_1", "from": "-30892#16", "to": "-31272#7"},
        {"id": "f_2", "from": "-31272#6", "to": "--30892#16"}, {"id": "f_3", "from": "--31272#7", "to": "--30892#16"},
        {"id": "f_4", "from": "--31272#7", "to": "--31272#6"}, {"id": "f_5", "from": "-30892#16", "to": "-30892#17"},
        {"id": "f_6", "from": "--31272#7", "to": "-30892#17"}, {"id": "f_7", "from": "--30892#17", "to": "--31272#6"},
        {"id": "f_8", "from": "-31272#6", "to": "--30892#16"}, {"id": "f_9", "from": "-30892#16", "to": "-31272#7"},
        {"id": "f_10", "from": "--31272#7", "to": "-30892#17"}, {"id": "f_11", "from": "-30892#16", "to": "--31272#6"},
        {"id": "f_12", "from": "-31272#6", "to": "-30892#17"}, {"id": "f_13", "from": "--30892#17", "to": "-31272#7"}
    ]

    for phase in TRAFFIC_PHASES:
        # ИСПРАВЛЕНИЕ: Убрали дефисы, оставили только текст фазы
        routes.append(ET.Comment(f" ФАЗА: {phase['name']} "))
        for edge in edges:
            flow = ET.SubElement(routes, "flow")
            flow.set("id", f"{edge['id']}_{phase['begin']}")
            flow.set("begin", str(phase['begin']))
            flow.set("end", str(phase['end']))
            prob = phase["probs"].get(edge["id"], phase["probs"]["default"])
            flow.set("probability", str(prob))
            flow.set("from", edge["from"])
            flow.set("to", edge["to"])

    xml_str = xml.dom.minidom.parseString(ET.tostring(routes)).toprettyxml(indent="    ")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(xml_str)
    
    print(f"✅ Трафик обновлен! Сгенерировано {len(TRAFFIC_PHASES)} фаз.")

if __name__ == "__main__":
    generate_dynamic_routes()