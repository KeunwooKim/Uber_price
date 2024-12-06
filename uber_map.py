import streamlit as st
import torch
from model import Model
from geopy.distance import geodesic
from streamlit_folium import st_folium
import folium

if 'click_count' not in st.session_state:
    st.session_state['click_count'] = 0
if 'start_coords' not in st.session_state:
    st.session_state['start_coords'] = None
if 'end_coords' not in st.session_state:
    st.session_state['end_coords'] = None

in_features = 14
hidden_features = [512, 512, 256, 256]
out_features = 1

model = Model(in_features=in_features, hidden_features=hidden_features, out_features=out_features)
model_file_path = './model/model.pth'
model.load_state_dict(torch.load(model_file_path))
model.eval()


def process_click(location):
    try:
        lat, lon = float(location['lat']), float(location['lng'])
        if st.session_state['click_count'] == 0:
            st.session_state['start_coords'] = (lat, lon)
            st.session_state['click_count'] = 1
        elif st.session_state['click_count'] == 1:
            st.session_state['end_coords'] = (lat, lon)
            st.session_state['click_count'] = 0
    except ValueError as e:
        st.error(f"Error processing the click: {e}")


def update_map():
    m = folium.Map(location=[42.3601, -71.0589], zoom_start=13)
    if st.session_state['start_coords']:
        folium.Marker(
            location=st.session_state['start_coords'],
            popup='Start',
            icon=folium.Icon(color='green')
        ).add_to(m)
    if st.session_state['end_coords']:
        folium.Marker(
            location=st.session_state['end_coords'],
            popup='End',
            icon=folium.Icon(color='red')
        ).add_to(m)
    return m


st.title("우버 가격 예측 서비스")
st.write("시작지점과 종료지점을 클릭으로 지정")

cab_type = st.selectbox("우버 차량 타입 선택",
                        ["UberX", "UberXL", "Lyft", "Lyft XL", "Shared", "Lux", "Lux Black", "Lux Black XL", "Black",
                         "Black SUV"])

cab_type_dict = {
    "UberX": 0,
    "UberXL": 1,
    "Lyft": 2,
    "Lyft XL": 3,
    "Shared": 4,
    "Lux": 5,
    "Lux Black": 6,
    "Lux Black XL": 7,
    "Black": 8,
    "Black SUV": 9
}

cab_type_encoded = cab_type_dict[cab_type]

st_data = st_folium(update_map(), width=700, height=450)

if 'last_clicked' in st_data and st_data['last_clicked']:
    process_click(st_data['last_clicked'])

if st.session_state['start_coords']:
    st.write(f"시작 좌표: {st.session_state['start_coords']}")
if st.session_state['end_coords']:
    st.write(f"종료 좌표: {st.session_state['end_coords']}")

if st.session_state['start_coords'] and st.session_state['end_coords']:
    distance = geodesic(st.session_state['start_coords'], st.session_state['end_coords']).miles
    st.write(f"거리: {distance} miles")


    input_features = [distance, cab_type_encoded] + [0] * (in_features - 2)


    if st.button("예측 하기"):
        input_tensor = torch.tensor([input_features], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor)
        st.write(f"예상 가격: ${prediction.item():.2f}")

if st.button("Reset"):
    st.session_state['start_coords'] = None
    st.session_state['end_coords'] = None
    st.session_state['click_count'] = 0
