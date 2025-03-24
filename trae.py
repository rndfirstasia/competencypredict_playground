#=============================================Library=====================================================
import streamlit as st
import pandas as pd
import boto3
from datetime import datetime
import mysql.connector
from mysql.connector import Error #untuk menunjukkan error yang terjadi saat menghubungkan apliasi ke database

from openai import OpenAI
import openai
import requests
import io #data akan disimpan di ram sementara, digunakan agar datanya lebih cepat dan tidak perlu disimpan di file yang akan lebih lambat
from io import StringIO #stringio digunakan untuk data string sedangkan bytesio digunakan untuk data biner
from collections import defaultdict
import pytz #datetime sesuai zona waktu indon

import time
from requests.adapters import HTTPAdapter #dari lib requests, untuk merubah jumlah maksimal percakapan atau durasi waktu tunggu
from urllib3.util.retry import Retry #retry untuk timeout, maksutnya jika request gagal karena alasan tertentu maka retry akan mencoba untuk mengulang permintaan
import re #regex

import json

import os
import google.generativeai as genai #generative ai dari google
#from google import genai
from pydub import AudioSegment
import tempfile

import jsonschema
from jsonschema import validate
from pydantic import BaseModel, ValidationError
import mimetypes

from typing_extensions import TypedDict, List

import asyncio
import aiohttp
import re
import time
from typing import List
from fastapi import UploadFile

import concurrent.futures

#=============================================Page Config=====================================================
st.set_page_config(
    page_icon="img/icon.png",
    page_title="Prediksi Kompetensi",
)

#=============================================ENV=====================================================
genai.configure(api_key=st.secrets['gemini']['api'])
aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
endpoint_url = st.secrets["aws"]["endpoint_url"]

mysql_user = st.secrets["mysql"]["username"]
mysql_password = st.secrets["mysql"]["password"]
mysql_host = st.secrets["mysql"]["host"]
mysql_port = st.secrets["mysql"]["port"]
mysql_database = st.secrets["mysql"]["database"]

openai.api_key=st.secrets["openai"]["api"]
client = OpenAI(api_key=st.secrets["openai"]["api"])
hf_token = st.secrets["hf"]["token"]
flask_url = st.secrets["flask"]["url"]

pito_url = st.secrets["sistem_fac"]["pito_url"]
vast_url = st.secrets["sistem_fac"]["vast_url"]
pito_api_user = st.secrets["sistem_fac"]["pito_api_user"]
pito_api_key = st.secrets["sistem_fac"]["pito_api_key"]
vast_api_user = st.secrets["sistem_fac"]["vast_api_user"]
vast_api_key = st.secrets["sistem_fac"]["vast_api_key"]

base_urls = {
    "PITO": pito_url,
    "VAST": vast_url
}

#=============================================Function=====================================================
#=============================================Function Tab 1=====================================================
def create_db_connection():
    try:
        conn = mysql.connector.connect(
            user=mysql_user,
            password=mysql_password,
            host=mysql_host,
            port=mysql_port,
            database=mysql_database
        )
        if conn.is_connected():
            return conn
        else:
            return None
    except Error as e:
        print(f"Error pada create_db_connection: {e}")
        return None
    
def get_levels_for_competency(id_competency):
    conn = create_db_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT level_name, level_description
        FROM pito_competency_level
        WHERE id_competency = %s
    """
    cursor.execute(query, (id_competency,))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    levels = [{"level_name": row[0], "level_description": row[1]} for row in results]
    return levels

def get_transcriptions(registration_id):
    conn = create_db_connection()
    if conn is None:
        st.error("Failed to connect to the database.")
        return []

    try:
        cursor = conn.cursor()
        query = """
        SELECT t.id_transkrip, t.registration_id, t.transkrip, t.speaker, t.start_section, t.end_section, a.num_speakers
        FROM txtan_transkrip t
        INNER JOIN txtan_audio a ON t.id_audio = a.id_audio
        WHERE t.registration_id = %s
        """
        cursor.execute(query, (1, registration_id))
        result = cursor.fetchall()
        #st.write(f"Transcriptions fetched: {len(result)}") #debug
        return result

    except Exception as e:
        st.error(f"Transcriptions fetched: {len(result)} for registration_id {registration_id}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

#menyimpan hasil pemisah antar pembicara ke tabel separator
def insert_into_separator(id_transkrip, registration_id, revisi_transkrip, revisi_speaker, revisi_start_section, revisi_end_section):
    conn = create_db_connection()
    if conn is None:
        st.error("Database connection failed.")
        return

    try:
        cursor = conn.cursor()
        query = """
        INSERT INTO txtan_separator (id_transkrip, registration_id, revisi_transkrip, revisi_speaker, revisi_start_section, revisi_end_section)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (id_transkrip, registration_id, revisi_transkrip, revisi_speaker, revisi_start_section, revisi_end_section))
        conn.commit()
    except mysql.connector.Error as e:
        st.error(f"Database insertion error: {e}")
    finally:
        conn.commit()
        cursor.close()
        conn.close()

    #st.write("Inserting into txtan_separator with values:", (id_transkrip, registration_id, revisi_transkrip, revisi_speaker, revisi_start_section, revisi_end_section)) #debug    

#fungsi untuk mengambil transkrip
def get_transcriptions(registration_id):
    conn = create_db_connection()
    if conn is None:
        st.error("Failed to connect to the database.")
        return []

    try:
        cursor = conn.cursor()
        query = """
        SELECT t.id_transkrip, t.registration_id, t.transkrip, t.speaker, t.start_section, t.end_section, a.num_speakers
        FROM txtan_transkrip t
        INNER JOIN txtan_audio a ON t.id_audio = a.id_audio
        WHERE t.registration_id = %s
        """
        # query = """
        # SELECT t.id_transkrip, t.registration_id, t.transkrip, t.speaker, t.start_section, t.end_section, a.num_speakers
        # FROM txtan_transkrip t
        # INNER JOIN txtan_audio a ON t.id_audio = a.id_audio
        # WHERE a.is_transcribed = %s AND t.registration_id = %s
        # """
        # st.write(f"Executing query: {query}") #debug
        cursor.execute(query, (1, registration_id))
        result = cursor.fetchall()
        #st.write(f"Transcriptions fetched: {len(result)}") #debug
        return result

    except Exception as e:
        st.error(f"Transcriptions fetched: {len(result)} for registration_id {registration_id}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

#fungsi untuk menyimpan hasil prediksi ke dalam tabel hasil kompetensi
def insert_into_result(final_predictions_df, registration_id):
    conn = create_db_connection()
    cursor = conn.cursor()
    query = """ 
    INSERT INTO txtan_competency_result (registration_id, competency, level, reason)
    VALUES (%s, %s, %s, %s)
    """

    for index, row in final_predictions_df.iterrows():
        competency = row['Kompetensi']
        level = row['Level']
        reason = row['Alasan Kemunculan']

        values = (registration_id, competency, level, reason)
        cursor.execute(query, values)

    # st.write("Inserting into txtan_separator with values:", (registration_id, competency, level, reason)) #debug

    conn.commit()
    cursor.close()
    conn.close()

    st.success("Step 5/5: Prediksi dibuat, proses selesai.") #debug

#fungsi untuk mengambil transkrip
#tapi ini belum menggunakan is_transcribed = 1
def get_transcriptions(registration_id):
    conn = create_db_connection()
    if conn is None:
        st.error("Failed to connect to the database.")
        return []

    try:
        cursor = conn.cursor()
        query = """
        SELECT t.id_transkrip, t.registration_id, t.transkrip, t.speaker, t.start_section, t.end_section, a.num_speakers
        FROM txtan_transkrip t
        INNER JOIN txtan_audio a ON t.id_audio = a.id_audio
        WHERE t.registration_id = %s
        """
        # query = """
        # SELECT t.id_transkrip, t.registration_id, t.transkrip, t.speaker, t.start_section, t.end_section, a.num_speakers
        # FROM txtan_transkrip t
        # INNER JOIN txtan_audio a ON t.id_audio = a.id_audio
        # WHERE a.is_transcribed = %s AND t.registration_id = %s
        # """
        # st.write(f"Executing query: {query}") #debug
        cursor.execute(query, (1, registration_id))
        result = cursor.fetchall()
        #st.write(f"Transcriptions fetched: {len(result)}") #debug
        return result

    except Exception as e:
        st.error(f"Transcriptions fetched: {len(result)} for registration_id {registration_id}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

#fungsi untuk menyimpan ke table separator
def insert_into_separator(id_transkrip, registration_id, revisi_transkrip, revisi_speaker, revisi_start_section, revisi_end_section):
    conn = create_db_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO txtan_separator (id_transkrip, registration_id, revisi_transkrip, revisi_speaker, revisi_start_section, revisi_end_section)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    values = (id_transkrip, registration_id, revisi_transkrip, revisi_speaker, revisi_start_section, revisi_end_section)
    cursor.execute(query, values)

    #st.write("Inserting into txtan_separator with values:", (id_transkrip, registration_id, revisi_transkrip, revisi_speaker, revisi_start_section, revisi_end_section)) #debug

    conn.commit()
    cursor.close()
    conn.close()

# Fungsi untuk mengoreksi label pembicara
def correct_speaker_labels(transkrip, num_speakers):
    prompt = (
        f"Berikut adalah transkrip dari percakapan interview dari {num_speakers} orang: \n"
        f"{transkrip}\n\n"
        "Dalam transkrip itu masih terdapat overlap antara Kandidat dan Assessor.\n"
        "Maka masukkan bagian yang overlap ke pembicara yang sebenarnya. Sehingga akan ada tanya jawab antar Assessor dan kandidat dan PASTI tidak hanya menjadi satu row.\n "
        "Jika orang lebih dari 2 maka akan ada lebih dari satu assessor. Kandidat tetap hanya akan ada satu.\n"
        "1. Kandidat (yang menjawab pertanyaan)\n"
        "2. Assessor (yang mengajukan pertanyaan)\n"
        "Contoh format dari bagian percakapan assessor dan kandidat:\n"
        "**Kandidat:** Untuk, misalkan contoh produknya ini sudah kita ekspor. Terus sudah kita coba untuk ekspor ke beberapa tempat, bagaimana supaya manajemen distribusinya (MD) itu produk ini dijalankan. Sudah kita ekspor, kita sesuaikan dengan promo yang mereka dari MD berikan. Karena kalau promonya tidak disesuaikan, secara otomatis produk ini nanti tidak akan terjual.\n"
        "**Assessor:** Kemudian, kalau dari sisi improvement, selama dua tahun terakhir ini boleh diceritakan seperti apa langkah improvement yang sudah pernah Bapak coba lakukan dan apakah inisiasinya dari diri Bapak sendiri? Ada contohnya seperti apa? Jika improvement terlalu banyak, seperti yang saya sampaikan tadi, karena kita lebih banyak, kalau saya sendiri.\n"
        "**Kandidat:** Kita lebih banyak ke ATM. Misalkan ada tim di tempat lain melakukan sesuatu, kita coba lakukan itu dengan sedikit modifikasi. Contohnya, kita selalu mengadakan yang namanya Red Light Promo. Itu salah satu usaha yang kita lakukan. Memang itu bukan gagasan dari saya, tapi gagasan dari beberapa toko. Tapi konsistensinya itu saya jalankan di tempat sini, konsistensi sebagaimana kita di tengah kondisi saat ini, contoh, trafik yang turun dan lain-lain, untuk menarik pelanggan yang datang ke toko, baik yang dari mal maupun yang dari luar. Itu yang saya konsistensikan dilakukan di toko ini.\n"
        "**Assessor:** Dengan melihat yang sudah dilakukan di toko-toko lain, jadi coba tetap konsisten dilakukan di tempat saat ini. Kalau misalkan dengan kondisi cabang saat ini, boleh diceritakan?.\n"
        "dan seterusnya.\n"
        "Tolong pastikan urutan dialog tetap seperti dalam transkrip asli, meskipun ada beberapa assessor.\n"
        "Betulkan juga bagian yang ada salah ketik atau ejaan yang kurang benar kecuali nama orang, nama perusahaan, nama jalan, nama kota, nama provinsi, nama negara, nama produk, singkatan.\n"
    )

    messages=[
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
        ],
    }
    ]

    try:
        # st.write("Sending request to API...") #debug
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0
        )

        # st.write("API Response:", response) #debug

        # Validasi respons dari API
        corrected_transcript = response.choices[0].message.content.strip()
        return corrected_transcript
        
    except Exception as e:
        st.error(f"Error while processing: {str(e)}")
        return None

def process_gpt_response_to_dataframe(gpt_response):

    try:
        json_data = gpt_response if isinstance(gpt_response, dict) else json.loads(gpt_response)

        transkrip_obj = TranscriptResponse(**json_data)

        df = pd.DataFrame([entry.dict() for entry in transkrip_obj.transkrip])
        df["registration_id"] = json_data.get("registration_id", "UNKNOWN")

        return df

    except json.JSONDecodeError:
        print("Error: Respons dari GPT bukan JSON yang valid")
        return None
    except Exception as err:
        print(f"Error: Data tidak valid menurut JSON Schema - {err}")
        return None

# Fungsi untuk memproses transkripsi
def process_transcriptions(registration_id):
    transcriptions = get_transcriptions(registration_id)
    # st.write(transcriptions) #debug

    if not transcriptions:
        st.error("No transcriptions found.")
        return
    
    transcriptions_by_registration = {}

    for transcription in transcriptions:
        reg_id = transcription[1]
        if reg_id not in transcriptions_by_registration:
            transcriptions_by_registration[reg_id] = []
        transcriptions_by_registration[reg_id].append(transcription)

    for registration_id, transcription_group in transcriptions_by_registration.items():
        combined_transcript = "\n".join([f"{t[3]}: {t[2]}" for t in transcription_group])
        num_speakers = transcription_group[0][6]

        #st.write(f"Processing transcription for registration_id {registration_id}")  #debug
        #st.write(combined_transcript) #debug

        corrected_transcript = correct_speaker_labels(combined_transcript, num_speakers)
        #st.write(f"Corrected Transcript: {corrected_transcript}") #debug
        if not corrected_transcript:
            st.error(f"Corrected Transcript is None for registration_id {registration_id}")
            continue

        df = process_gpt_response_to_dataframe(corrected_transcript)
        #st.write(df) #debug
        
        if df.empty:
            st.error(f"Empty DataFrame for registration_id {registration_id}.")
            continue
        
        #st.write(f"Processed DataFrame for {registration_id}:", df)  #debug

        # Merger text dan speaker
        merged_text = []
        merged_speakers = []
        previous_speaker = None
        temp_text = ""
        temp_speaker = ""

        for _, row in df.iterrows():
            current_speaker = row['speaker']
            current_text = row['text']

            if current_speaker == previous_speaker:
                temp_text += ' ' + current_text
            else:
                if previous_speaker is not None:
                    merged_text.append(temp_text)
                    merged_speakers.append(temp_speaker)
                
                temp_text = current_text
                temp_speaker = current_speaker
                previous_speaker = current_speaker

        if temp_text:
            merged_text.append(temp_text)
            merged_speakers.append(temp_speaker)

        df_merged = pd.DataFrame({
            'text': merged_text,
            'speaker': merged_speakers
        })

        df_merged['text'] = df_merged['text'].replace(r'\s+', ' ', regex=True)

        for index, row in df_merged.iterrows():
            #st.write(f"Inserting into txtan_separator: {row['text']}, {row['speaker']}") #debug
            insert_into_separator(
                transcription_group[0][0], 
                registration_id, 
                row['text'], 
                row['speaker'], 
                transcription_group[0][4], 
                transcription_group[0][5]
            )

        #st.success("Transcriptions processed and inserted.") #debug

def update_transcription_status(id_audio):
    conn = create_db_connection()

    try:
        cursor = conn.cursor()

        update_query = '''
            UPDATE txtan_audio
            SET is_transcribed = 1
            WHERE id_audio = %s
        '''
        cursor.execute(update_query, (id_audio,))
        conn.commit()
        print(f"Audio with id_audio {id_audio} marked as transcribed.")

    except Exception as e:
        print(f"Error: {e}")

def get_separator(registration_id):
    conn = create_db_connection()
    cursor = conn.cursor()
    query = """
    SELECT s.id_transkrip, s.registration_id, s.revisi_transkrip, s.revisi_speaker, s.revisi_start_section, s.revisi_end_section
    FROM txtan_separator s
    INNER JOIN txtan_audio a ON s.registration_id = a.registration_id
    WHERE s.registration_id = %s
    """

    cursor.execute(query, (registration_id,))
    result = cursor.fetchall()

    #st.write(f"Separator data fetched: {len(result)} entries for registration_id {registration_id}") #debug

    cursor.close()
    conn.close()
    return result            

def get_competency(registration_id):
    conn = create_db_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT
            prd.name_product,
            comp.competency,
            comp.description,
            lvl.level_value,
            lvl.level_name,
            lvl.level_description
        FROM txtan_audio a
        JOIN pito_product prd ON prd.id_product = a.id_product
        JOIN pito_competency comp ON comp.id_product = prd.id_product
        LEFT JOIN pito_competency_level lvl ON lvl.id_competency = comp.id_competency
        WHERE a.registration_id = %s
    """
    
    cursor.execute(query, (registration_id,))
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    #st.write(f"hasil query competency: {competencies}")#debug
    
    # Kembalikan hasil sebagai daftar dictionary agar mudah digunakan
    competencies = [{
        "product": row[0],
        "competency": row[1],
        "description": row[2],
        "level_value": row[3],
        "level_name": row[4],
        "level_description": row[5]
    } for row in result]
    
    return competencies

def get_level_set_from_audio_table(registration_id):
    query = """
    SELECT a.id_level_set, lvl.name_level AS 'NAMA LEVEL'
    FROM txtan_audio a
    JOIN pito_level lvl ON a.id_level_set = lvl.id_level_set
    WHERE a.registration_id = %s
    """
    conn = create_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query, (registration_id,))
        result = cursor.fetchone()
        cursor.fetchall()
        return result if result else (None, None)
    except Exception as e:
        print(f"Error fetching level set: {e}")
        return None, None
    finally:
        cursor.close()
        conn.close()

def get_name_levels_from_id_level_set(id_level_set):
    conn = create_db_connection()
    cursor = conn.cursor()

    query = """
    SELECT name_level FROM pito_level
    WHERE id_level_set = %s
    """
    cursor.execute(query, (id_level_set,))
    result = cursor.fetchall()

    cursor.close()
    conn.close()

    name_levels = [row[0] for row in result]

    return name_levels

def predict_competency(combined_text, competencies, level_names):

    #st.write(f"Predict Competency ID Level Set: {id_level_set}") #debug

    #name_levels = get_name_levels_from_id_level_set(id_level_set)

    prompt = "Saya memiliki transkrip hasil dari wawancara dan daftar kompetensi yang ingin diidentifikasi.\n\n"
    prompt += "Buatlah hasil analisa menjadi bentuk tabel dan prediksi juga levelnya.\n"
    prompt += "Hasil yang dikeluarkan WAJIB table dan TANPA FORMAT TEXT bold, italic atau sejenisnya.\n"

    prompt += "header kolom table HARUS menggunakan huruf kapital di awal dan dikuti dengan huruf kecil\n"

    prompt += f"Gunakan hanya level dari daftar berikut: {', '.join(level_names)}.\n" ### ini name levelnya belum ada
    prompt += "Pastikan level yang digunakan sesuai dengan level yang dipilih dan WAJIB DALAM BAHASA INGGRIS.\n"
    
    #prompt += "Level yang digunakan sesuai yang tercantum dibawah, semisal ada level 1 sampai level 5 maka level 5 adalah paling besar, atau jika ada very low sampai very high maka very high adalah paling besar. dan level WAJIB dalam bahasa inggris.\n"
    #prompt += f"Level yang digunakan juga mengikuti dari {dropdown_options_predict_competency} dan level WAJIB dalam bahasa inggris.\n"
    prompt += f"Teks transkrip berikut: {combined_text}\n\n"
    prompt += "Berikut adalah daftar kompetensi dengan level dan deskripsinya:\n"
    
    for competency in competencies:
        prompt += (f"- Kompetensi Bernama: {competency['competency']} deskripsinya adalah\n")
        
        #kalau ada level
        if competency.get("levels"):
            prompt += "  Level:\n"
            for level in competency["levels"]:
                level_description = level["level_description"] if level["level_description"] else competency['description']
                prompt += (f"    - Name: {level['level_name']}\n"
                        f"      Deskripsi Level: {level_description}\n")
        else:
            prompt += f"  (Tidak ada level spesifik, gunakan deskripsi kompetensi umum: {competency['description']})\n"
            # prompt += "Level yang digunakan adalah Very High, High, Medium, Low, Very Low dan level WAJIB dalam bahasa inggris.\n"
            prompt += f" Serta level mengikuti dari {level_names}."

    prompt += "\nHasil hanya akan berupa tabel dengan kolom: Kompetensi, Level, dan Alasan Kemunculan\n"
    
    st.write(f"Prompt: {prompt}") #debug

    messages=[
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
        ],
    }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages = messages,
        temperature=0,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0
    )

    corrected_transcript_dict = response.model_dump()
    corrected_transcript = corrected_transcript_dict['choices'][0]['message']['content']
    return corrected_transcript

def combine_text_by_registration(separator_data):
    combined_data = defaultdict(lambda: {"revisi_transkrip": "", "revisi_speaker": ""})

    for record in separator_data:
        registration_id = record[1] #ini kuraang yakin harusnya dimulai dari 0 atau 1, nanti di cek
        revisi_transkrip = record[2] or ""
        revisi_speaker = record[3] or ""

        combined_data[registration_id]["revisi_transkrip"] += f" {revisi_transkrip}"
        combined_data[registration_id]["revisi_speaker"] += f" {revisi_speaker}"

    return combined_data

def predictor(registration_id, dropdown_options_predict_competency):
    separator_data = get_separator(registration_id) #mengambil data transkrip dari table txtan_transkrip
    #st.write(f"Separator data: {separator_data}") #debug
    competency_data = get_competency(registration_id) #mengambil data kompetensi
    #st.write(f"Competency data: {competency_data}") #debug

    #st.write(f"Fetched {len(separator_data)} separator data entries") #debug
    #st.write(f"Fetched {len(competency_data)} competency data entries") #debug

    if not separator_data:
        st.error("No data found in the separator table.")
        return

    if not competency_data:
        st.error("No competency data found.")
        return

    competency_list = [{"competency": row.get("competency"), 
                        "description": row.get("description"),
                        **({
                            "level_value": row.get("level_value"),
                            "level_name": row.get("level_name"),
                            "level_description": row.get("level_description")
                            }if row.get("level_value") and row.get("level_name") and row.get("level_description") else {})
                        } 
                        for row in competency_data]
    #st.write(f"Competency list: {competency_list}") #debug

    combined_data = combine_text_by_registration(separator_data)
    #st.write(f"combined_data: {combined_data}") #debug

    all_predictions = []

    for registration_id, text_data in combined_data.items():
        combined_text = f"{text_data['revisi_transkrip']} {text_data['revisi_speaker']}"

        # st.success(f"Step 4/5: Mohon tunggu, proses prediksi berlangsung.....") #debug
        
        predicted_competency = predict_competency(combined_text, competency_list, level_names)

        #st.write(f"Predicted competency for {registration_id}:\n{predicted_competency}") #debug

        try:
            df_competency = pd.read_csv(StringIO(predicted_competency), sep='|', skipinitialspace=True)
            df_competency.columns = df_competency.columns.str.strip()
            df_competency['registration_id'] = registration_id
            st.success(f"Step 4/5: Mohon tunggu, proses prediksi berlangsung.....") #debug

            all_predictions.append(df_competency)

        except Exception as e:
            st.error(f"Error processing prediction for registration ID {registration_id}: {e}")
    
    #st.write(all_predictions) #debug

    if all_predictions:
        #st.write(f"all_predictions before: {all_predictions}")  # debug
        
        if isinstance(all_predictions, list) and all(isinstance(df, pd.DataFrame) for df in all_predictions):
            final_predictions_df = pd.concat(all_predictions, ignore_index=True)
            #st.dataframe(f"Final pred CONCAT: {final_predictions_df}") #debug
            final_predictions_df = final_predictions_df.applymap(lambda x: x.replace('**', '') if isinstance(x, str) else x)
            #st.dataframe(f"Final pred MAP: {final_predictions_df}") #debug
            final_predictions_df = final_predictions_df.drop(index=0).reset_index(drop=True)
            #st.dataframe(f"Final pred DROP dan RESET INDEX: {final_predictions_df}") #debug
            
            #st.write(f"Final pred DONE: {final_predictions_df}")  # debug
            
            insert_into_result(final_predictions_df, registration_id)
        else:
            st.error("Error: all_predictions harus berupa list yang berisi DataFrame.")
    else:
        st.error("Error: all_predictions kosong.")

#ambil data hasil transkrip pada 
def fetch_transkrip_from_db(registration_id):
    conn = create_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT transkrip, speaker, start_section, end_section
    FROM txtan_transkrip
    WHERE registration_id = %s
    """
    cursor.execute(query, (registration_id,))
    transkrip_data = cursor.fetchall()

    cursor.close()
    conn.close()

    return transkrip_data

class NamedBytesIO(io.BytesIO):
    def __init__(self, content, name):
        super().__init__(content)
        self.name = name 

def transcribe_with_whisper(audio_file):
    if not audio_file:
        raise ValueError("File audio tidak diberikan")
    
    if hasattr(audio_file, 'name'):
        audio_file_name = audio_file.name
    else:
        raise ValueError("Objek audio tidak memiliki atribut nama file")

    st.write(f"Mengirim file ke Whisper API: {audio_file_name}")

    audio_bytes = audio_file.getvalue()
    
    audio_file_whisper = NamedBytesIO(audio_bytes, audio_file_name)

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=(audio_file_name, audio_file_whisper, "audio/m4a"),  
        response_format="text"
    )

    return transcript

def separate_speakers(transcript, num_speakers=2):
    prompt = f"""
    Berikut adalah transkrip wawancara dengan {num_speakers} orang.
    Pisahkan dialog berdasarkan peran:
    - **Kandidat** (yang menjawab pertanyaan)
    - **Assessor** (yang bertanya)
    
    Transkripsi: {transcript}
    
    Format keluaran:
    **Kandidat:** [isi dialog]
    **Assessor:** [isi dialog]
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def transcribe_with_whisper(audio_file):
    if not audio_file:
        raise ValueError("File audio tidak diberikan")
    
    if hasattr(audio_file, 'name'):
        audio_file_name = audio_file.name
    else:
        raise ValueError("Objek audio tidak memiliki atribut nama file")
    
    st.write(f"Mengirim file ke Whisper API: {audio_file_name}")
    
    audio_bytes = audio_file.getvalue()
    
    class NamedBytesIO(io.BytesIO):
        def __init__(self, content, name):
            super().__init__(content)
            self.name = name
    
    audio_file_whisper = NamedBytesIO(audio_bytes, audio_file_name)
    
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=(audio_file_name, audio_file_whisper, "audio/m4a"),
        response_format="text"
    )
    
    process_gpt_response_to_dataframe(transcript)
    st.success("Step 2/5: Audio berhasil dikirim untuk transkripsi.")
    return transcript

class TranskripItem(TypedDict):
    text: str
    speaker: str

class TranskripResponse(BaseModel):
    transkrip: List[TranskripItem]

class TranscriptEntry(BaseModel):
    text: str
    speaker: str

class TranscriptResponse(BaseModel):
    transkrip: list[TranscriptEntry]

def flatten_schema(schema):
    if "$defs" in schema:
        definitions = schema.pop("$defs")

        def replace_refs(obj):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_key = obj["$ref"].split("/")[-1]
                    obj.clear()
                    obj.update(definitions[ref_key])
                else:
                    for key, value in obj.items():
                        replace_refs(value)
            elif isinstance(obj, list):
                for item in obj:
                    replace_refs(item)

        replace_refs(schema)

    return schema

#transcript_schema = TranscriptResponse.model_json_schema(by_alias=True, ref_template='{model}Schema')
#transcript_schema = TranscriptResponse.model_json_schema(by_alias=True)
transcript_schema = {
    "type": "object",
    "properties": {
        "transkrip": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "speaker": {
                        "type": "string",
                        "enum": ["Assessor", "Kandidat"]  
                    }
                },
                "required": ["text", "speaker"]
            }
        }
    },
    "required": ["transkrip"]
}

transcript_schema = flatten_schema(transcript_schema)

#remove titles untuk kompabilitas dengan gemini API
def remove_titles(schema):
    if isinstance(schema, dict):
        schema.pop("title", None)
        for value in schema.values():
            remove_titles(value)
    elif isinstance(schema, list):
        for item in schema:
            remove_titles(item)

remove_titles(transcript_schema)

final_schema = {
    "type": "object",
    "properties": transcript_schema["properties"]
}

#melakukan validasi kepada json yang diharapkan
def validate_json(data):
    if not isinstance(data, dict):
        return False
    if "transkrip" not in data or not isinstance(data["transkrip"], list):
        return False
    for item in data["transkrip"]:
        if not isinstance(item, dict) or "text" not in item or "speaker" not in item:
            return False
        if not isinstance(item["text"], str) or not isinstance(item["speaker"], str):
            return False
    return True

#ini kan split, nah kalo misal di split terus ada yang kurang tinggal di recover
def recover_partial_json(text):
    try:
        match = re.search(r'{\s*"transkrip"\s*:\s*\[\s*({.*})\s*\]', text, re.DOTALL)
        if match:
            entries_text = f'[{match.group(1)}]'
            try:
                entries = json.loads(entries_text)
                valid_entries = []
                for entry in entries:
                    if 'speaker' in entry and 'text' in entry:
                        valid_entries.append(entry)
                return valid_entries
            except:
                pass
        
        entries = []
        pattern = r'{\s*"speaker"\s*:\s*"(Assessor|Kandidat)"\s*,\s*"text"\s*:\s*"([^"]*)"'
        for match in re.finditer(pattern, text):
            speaker = match.group(1)
            text_content = match.group(2)
            entries.append({"speaker": speaker, "text": text_content})
        
        return entries if entries else None
    except Exception as e:
        st.warning(f"Error during JSON recovery: {e}")
        return None

def process_chunk(chunk_file, model, chunk_number, total_chunks, max_retries=3):
    for retry in range(max_retries):
        try:
            with open(chunk_file.name, "rb") as f:
                uploaded_file = genai.upload_file(f, mime_type="audio/wav")
                
                response = model.generate_content(
                    uploaded_file,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0,
                        "max_output_tokens": 1048576,
                        "response_schema": transcript_schema
                    }
                )
                
                chunk_text = "".join(resp.text for resp in response)
                
                try:
                    chunk_json = json.loads(chunk_text)
                    if validate_json(chunk_json):
                        st.success(f"Chunk {chunk_number}/{total_chunks}: Successfully transcribed")
                        return chunk_json["transkrip"]
                    else:
                        st.warning(f"Chunk {chunk_number}/{total_chunks}: Invalid JSON structure")
                        recovered_data = recover_partial_json(chunk_text)
                        if recovered_data:
                            return recovered_data
                except json.JSONDecodeError:
                    if retry < max_retries - 1:
                        st.warning(f"Retry {retry + 1}/{max_retries} for chunk {chunk_number}")
                        time.sleep(2)  # Add delay between retries
                        continue
                    st.error(f"Chunk {chunk_number}/{total_chunks}: JSON decode error")
                    recovered_data = recover_partial_json(chunk_text)
                    if recovered_data:
                        return recovered_data
        except Exception as e:
            if retry < max_retries - 1:
                st.warning(f"Retry {retry + 1}/{max_retries} for chunk {chunk_number} due to error: {str(e)}")
                time.sleep(2)  # Add delay between retries
                continue
            st.error(f"Error processing chunk {chunk_number}: {e}")
            
            # Try processing in smaller chunks as a last resort
            if len(chunk) > 120000:
                return process_smaller_chunks(chunk, model, chunk_number, total_chunks)
    return None

#melakukan transkrip pada dengan gemini
def transcribe_audio_gemini(audio_file, file_extension, id_input_id_kandidat):
    try:
        # Create temporary directory that will be automatically cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare audio file
            audio_file.seek(0)
            temp_file_path = os.path.join(temp_dir, f"input.{file_extension}")
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(audio_file.read())
            
            # Load audio using pydub
            audio = AudioSegment.from_file(temp_file_path, format=file_extension)
            
            # Split into 5-minute chunks (300000ms)
            chunk_length_ms = 300000
            chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            
            st.write(f"Split audio into {len(chunks)} chunks of 5 minutes each")
            
            model = genai.GenerativeModel("gemini-2.0-flash")
            chunk_results = {}
            
            # Process chunks concurrently with max 4 workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
                future_to_chunk = {}
                
                # Submit all chunks for processing
                for i, chunk in enumerate(chunks):
                    chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                    chunk.export(chunk_path, format="wav")
                    
                    with open(chunk_path, "rb") as f:
                        future = executor.submit(
                            process_chunk,
                            tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=temp_dir),
                            model,
                            i+1,
                            len(chunks)
                        )
                        future_to_chunk[future] = i
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_index = future_to_chunk[future]
                    try:
                        result = future.result()
                        if result:
                            chunk_results[chunk_index] = result
                            st.success(f"Chunk {chunk_index + 1}/{len(chunks)}: Successfully processed")
                    except Exception as e:
                        st.error(f"Error processing chunk {chunk_index + 1}: {e}")
            
            # Combine results in correct order
            all_transcript_entries = []
            for i in range(len(chunks)):
                if i in chunk_results:
                    all_transcript_entries.extend(chunk_results[i])
            
            final_result = {
                "transkrip": all_transcript_entries
            }
            
            # Validate and return final result
            try:
                transkrip_json = TranscriptResponse(**final_result)
                return {
                    "registration_id": id_input_id_kandidat,
                    "transkrip": transkrip_json.model_dump()
                }
            except ValidationError as e:
                st.error("Error: Final combined output doesn't match JSON Schema.")
                st.write(e.json())
                return None

    except Exception as e:
        st.error(f"Error during Gemini API transcription: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None
    
def insert_into_separator(id_transkrip, registration_id, revisi_transkrip, revisi_speaker, revisi_start_section, revisi_end_section):
    conn = create_db_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO txtan_separator (id_transkrip, registration_id, revisi_transkrip, revisi_speaker, revisi_start_section, revisi_end_section)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    values = (id_transkrip, registration_id, revisi_transkrip, revisi_speaker, revisi_start_section, revisi_end_section)
    cursor.execute(query, values)

    conn.commit()
    cursor.close()
    conn.close()

#upload audio ke boto3
def upload_audio_to_s3(audio_file, file_name):
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
            endpoint_url=st.secrets["aws"]["endpoint_url"]
        )

        bucket_name = 'rpi-ta'
        s3_client.upload_fileobj(audio_file, bucket_name, file_name)

        return f"s3://{bucket_name}/{file_name}"
    except Exception as e:
        st.error(f"Error uploading to S3: {e}")
        return None

#=============================================Function Tab 2=====================================================
def get_transkrip_data(registration_id):
    conn = create_db_connection()
    if conn is None:
        st.error("Database connection not available.")
        return pd.DataFrame(columns=["Start", "End", "Transkrip", "Speaker"])

    try:
        cursor = conn.cursor()
        query = """
        SELECT revisi_start_section AS 'Start', revisi_end_section AS 'End', revisi_transkrip AS 'Transkrip', revisi_speaker AS 'Speaker'
        FROM txtan_separator
        WHERE registration_id = %s
        """
        cursor.execute(query, (registration_id,))
        result = cursor.fetchall()
        cursor.close()
        conn.close()

        if result:
            df = pd.DataFrame(result, columns=["Start", "End", "Transkrip", "Speaker"]) #start dan end masihh dalam sec
            return df
        else:
            return pd.DataFrame(columns=["Start", "End", "Transkrip", "Speaker"])

    except mysql.connector.Error as e:
        st.error(f"Error fetching transcription data: {e}")
        return pd.DataFrame(columns=["Start", "End", "Transkrip", "Speaker"])
    finally:
        if conn.is_connected():
            conn.close()

#=============================================Function Tab 3=====================================================
def get_result_data(registration_id):
    query = """
    SELECT competency, level, reason
    FROM txtan_competency_result
    WHERE registration_id = %s
    """
    conn = create_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, (registration_id,))
    result = cursor.fetchall()

    cursor.close()

    if result:
        df = pd.DataFrame(result, columns=["competency", "level", "reason"])
        return df
    else:
        return pd.DataFrame(columns=["competency", "level", "reason"])

def get_all_so_values(registration_id):
    conn = create_db_connection()
    try:
        cursor = conn.cursor()
        query = """
        SELECT competency, so_level, so_reason
        FROM txtan_competency_result
        WHERE registration_id = %s
        """
        cursor.execute(query, (registration_id,))
        return cursor.fetchall() 
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return []  
    finally:
        cursor.close()
        conn.close()

def update_single_entry_db(conn, competency, level, reason, so_level, so_reason, registration_id):
    try:
        cursor = conn.cursor()

        check_query = """
        SELECT COUNT(*) FROM txtan_competency_result
        WHERE registration_id = %s AND competency = %s AND level = %s AND reason = %s
        """
        cursor.execute(check_query, (registration_id, competency, level, reason))
        count = cursor.fetchone()[0]

        if count > 0:
            update_query = """
            UPDATE txtan_competency_result
            SET so_level = %s, so_reason = %s
            WHERE registration_id = %s AND competency = %s AND level = %s AND reason = %s
            """
            cursor.execute(update_query, (so_level, so_reason, registration_id, competency, level, reason))
        else:
            insert_query = """
            INSERT INTO txtan_competency_result (registration_id, competency, level, reason, so_level, so_reason)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (registration_id, competency, level, reason, so_level, so_reason))

        conn.commit()
        # st.write("Changes committed to the database.")  # Commented out for production
    except Exception as e:
        print(f"Error updating or inserting entry: {e}")
    finally:
        cursor.close()

#=============================================Function Tab 4=====================================================
#=============================================Function Subtab 1=====================================================
def save_competencies_to_db(id_product):
    conn = create_db_connection()
    cursor = conn.cursor()

    query_find_competency = """
        SELECT id_competency FROM pito_competency WHERE competency = %s
    """
    query_insert_competency = """
        INSERT INTO pito_competency (id_product, competency, description) 
        VALUES (%s, %s, %s)
    """
    query_find_level = """
        SELECT id_pito_competency_level FROM pito_competency_level 
        WHERE id_competency = %s AND level_value = %s
    """
    query_insert_level = """
        INSERT INTO pito_competency_level (level_value, level_name, level_description, id_competency) 
        VALUES (%s, %s, %s, %s)
    """

    for competency, description, levels in st.session_state['competencies']:
        cursor.execute(query_find_competency, (competency,))
        result = cursor.fetchone()

        if result:
            id_competency = result[0]
        else:
            cursor.execute(query_insert_competency, (id_product, competency, description))
            conn.commit()
            id_competency = cursor.lastrowid  

        for level in levels:
            cursor.execute(query_find_level, (id_competency, level["value"]))
            level_exists = cursor.fetchone()

            if not level_exists:
                cursor.execute(query_insert_level, (
                    level["value"],
                    level["name"],
                    level["description"],
                    id_competency
                ))
            else:
                st.warning(f"Level Value '{level['value']}' sudah ada untuk kompetensi '{competency}' dan tidak akan ditambahkan lagi.")

    conn.commit()
    cursor.close()
    conn.close()

def is_product_exists(product_name):
    conn = create_db_connection()
    cursor = conn.cursor()
    
    query_check = """
        SELECT COUNT(*) FROM pito_product WHERE name_product = %s
    """
    cursor.execute(query_check, (product_name,))
    exists = cursor.fetchone()[0] > 0
    
    cursor.close()
    conn.close()
    
    return exists

#=============================================Function Subtab 2=====================================================
def save_level_set_to_db(level_set_name, levels_name, levels_value):
    conn = create_db_connection()
    cursor = conn.cursor()

    try:
        query_check_existing = """
            SELECT COUNT(*)
            FROM pito_level 
            WHERE id_level_set = %s
        """

        cursor.execute(query_check_existing, (level_set_name,))
        existing_count = cursor.fetchone()[0]

        if existing_count > 0:
            st.error(f"{level_set_name} sudah ada, mohon gunakan nama lain")
            return

        query_insert_level = """
            INSERT INTO pito_level (name_level, value_level, id_level_set)
            VALUES (%s, %s, %s)
        """
        for name, value in zip(levels_name, levels_value):
            cursor.execute(query_insert_level, (name, value, level_set_name))
        
        conn.commit()
    
    except Exception as e:
        st.error(f"Error saat menyimpan level set: {e}")
    
    finally:
        cursor.close()
        conn.close()

def get_existing_levels(level_set_name):
    conn = create_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT name_level, value_level
        FROM pito_level
        WHERE id_level_set = %s
    """

    cursor.execute(query, (level_set_name,))
    result = cursor.fetchall()
    cursor.close()
    conn.close()

    return result

#=============================================Function Subtab 3=====================================================
def get_existing_assessor(assessor_code):
    conn = create_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT kode_assessor, name_assessor
        FROM txtan_assessor
        WHERE kode_assessor = %s
    """

    cursor.execute(query, (assessor_code,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    return result

def save_assessor_to_db(assessor_code, name_assessor):
    conn = create_db_connection()
    cursor = conn.cursor()

    try:
        existing_assessor = get_existing_assessor(assessor_code)

        if existing_assessor:
            existing_name_assessor = existing_assessor[1]
            st.error(f'Assessor dengan kode {assessor_code} sudah digunakan oleh {existing_name_assessor}, mohon gunakan kode lain.')
            return

        query_insert_assessor = """
        INSERT INTO txtan_assessor (kode_assessor, name_assessor)
        VALUES (%s, %s)
        """
        cursor.execute(query_insert_assessor, (assessor_code, name_assessor))
        conn.commit()
        st.success(f"Assessor {name_assessor} dengan kode {assessor_code} berhasil disimpan")

    except Exception as e:
        st.error(f"Error saat menyimpan kode assessor: {e}")

    finally:
        cursor.close()
        conn.close()

#=======================================================================================================
#=============================================BODY======================================================
#=======================================================================================================

conn = create_db_connection()

if conn:
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM txtan_assessor;')
    df_txtan_assessor = cursor.fetchall()
    column_name_txtan_assessor = [i[0] for i in cursor.description]
    df_txtan_assessor = pd.DataFrame(df_txtan_assessor, columns=column_name_txtan_assessor)

    cursor.execute("""
    SELECT
        pdc.id_product,                          
        pdc.name_product AS 'PRODUCT',
        comp.competency AS 'COMPETENCY',
        comp.description AS 'COMPETENCY DESCRIPTION',
        lvl.level_name AS 'LEVEL NAME',
        lvl.level_description AS 'LEVEL DESCRIPTION',
        comp.id_competency AS 'id_competency'
    FROM `pito_product` AS pdc
    JOIN pito_competency AS comp ON comp.id_product = pdc.id_product
    LEFT JOIN pito_competency_level AS lvl ON comp.id_competency = lvl.id_competency
    """)
    df_pito_product = cursor.fetchall()
    column_names_pito_product = [i[0] for i in cursor.description]
    df_pito_product = pd.DataFrame(df_pito_product, columns=column_names_pito_product)
    options_product_set = [""] + df_pito_product['PRODUCT'].drop_duplicates().tolist() #list produk dari database

    cursor.execute("""
    SELECT
        lvl.name_level AS 'NAMA LEVEL',
        lvl.value_level,
        lvl.id_level_set
    FROM pito_level AS lvl;
    """)
    df_pito_level = cursor.fetchall()
    column_names_pito_level = [i[0] for i in cursor.description]
    df_pito_level = pd.DataFrame(df_pito_level, columns=column_names_pito_level)
    df_pito_level['id_level_set'] = df_pito_level['id_level_set'].astype(str)

    #st.dataframe(df_pito_level) #debug

    options_level_set = [""] + df_pito_level['id_level_set'].drop_duplicates().tolist() #list level dari database
    #st.write(f"df pito level: {options_level_set}") #debug
    cursor.close()
    conn.close()
else:
    st.error("Tidak bisa terhubung ke database")

st.header("Aplikasi Prediksi Kompetensi")

# Sidebar for navigation
st.sidebar.title("Parameter")
options_num_speaker = [ '2', '1', '3', '4', '5', '6']

#Sidebar
id_input_kode_assessor = st.sidebar.text_input("Kode Assessor Anda")
id_input_id_kandidat = st.sidebar.text_input("ID Kandidat")
selected_base_url = st.sidebar.selectbox("Pilih Sistem:", list(base_urls.keys()))
selected_option_num_speaker = st.sidebar.selectbox("Jumlah Speaker", options_num_speaker)
selected_option_product_set = st.sidebar.selectbox("Set Kompetensi", options_product_set)
selected_option_level_set = st.sidebar.selectbox("Set Level", options_level_set)
        
#connect API kandidat dengan PITO
if id_input_id_kandidat:
    headers = {
        "PITO": {
            "X-API-USER": pito_api_user,
            "X-API-KEY": pito_api_key
        },
        "VAST": {
            "X-API-USER": vast_api_user,
            "X-API-KEY": vast_api_key
        }
    }

    base_url = base_urls[selected_base_url]
    url = f"{base_url}{id_input_id_kandidat}"
    selected_headers = headers[selected_base_url]

    # max_retires = 5
    # for attempt in range(max_retires):
    response_id_kandidat = requests.get(url, headers=selected_headers)
    #cek apakah request sukses
    if response_id_kandidat.status_code == 200:
        try:
            api_data = response_id_kandidat.json()
            st.session_state.response_id_kandidat = api_data
        except Exception as e:
            st.write(f"Error info id kandidat: {e}")
            
        api_id_kandidat = api_data["data"].get('id', 'Tidak tersedia')
        api_nama = api_data["data"].get('name', 'Tidak tersedia')
        api_jenis_kelamin = api_data["data"].get('gender', 'Tidak tersedia')
        api_produk = api_data["data"].get('product', 'Tidak tersedia')
        api_client = api_data["data"].get('client', 'Tidak tersedia')
        api_dob = api_data["data"].get('dob', 'Tidak tersedia')

        #st.write(response_id_kandidat.text) #debug
        
        with st.container(border=True):
            st.write("#### Informasi ID Kandidat")
            
            st.write(f"ID Kandidat: {api_id_kandidat}")
            st.write(f"Nama: {api_nama}")
            st.write(f"Tanggal Lahir: {api_dob}")
            st.write(f"Jenis Kelamin: {api_jenis_kelamin}")
            st.write(f"Klien: {api_client}")
            st.write(f"Produk: {api_produk}")
        
    else:
        st.error(f"ID Kandidat tidak terdaftar/Sistem salah")
else:
    st.warning("Silakan masukkan ID Kandidat.")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Input Informasi", "📄 Hasil Transkrip", "🖨️ Hasil Prediksi", "⚙️ <admin> Input"])

#=======================================================================================================
#=============================================TAB 1=====================================================
#=======================================================================================================
with tab1:
    if not id_input_kode_assessor: #setting default kalau tidak ada kode assessor
        st.warning("Mohon masukkan kode Assessor Anda.")
    else:
        assessor_row = df_txtan_assessor[df_txtan_assessor['kode_assessor'].str.lower() == id_input_kode_assessor.lower()] #kode assessor bisa besar atau kecil

        if not assessor_row.empty:
            nama_assessor = assessor_row['name_assessor'].values[0]
            st.subheader(f"Selamat Datang, {nama_assessor}")
        else:
            st.subheader("Kode Assessor tidak terdaftar.")

    selected_product = df_pito_product[df_pito_product["PRODUCT"] == selected_option_product_set]
    with st.container(border=True):
        def get_levels_for_competency(id_competency):
            conn = create_db_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT level_name, level_description
                FROM pito_competency_level
                WHERE id_competency = %s
            """
            cursor.execute(query, (id_competency,))
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            levels = [{"level_name": row[0], "level_description": row[1]} for row in results]
            return levels
        
        competency_data = {}
        for _, row in df_pito_product.iterrows():
            if row['PRODUCT'] == selected_option_product_set:
                id_competency = row['id_competency']
                
                if id_competency not in competency_data:
                    competency_data[id_competency] = {
                        "product": row['PRODUCT'],
                        "competency": row['COMPETENCY'],
                        "description": row['COMPETENCY DESCRIPTION'],
                        "levels": []
                    }
                
                if row['LEVEL NAME'] and row['LEVEL DESCRIPTION']:
                    competency_data[id_competency]["levels"].append({
                        "level_name": row['LEVEL NAME'],
                        "level_description": row['LEVEL DESCRIPTION']
                    })

        competency_list = list(competency_data.values())
        
        if not selected_option_product_set:
            st.warning("Silahkan pilih set kompetensi")
        else:
            st.write(f'#### Set Kompetensi dari {selected_option_product_set}')
            if competency_list:
                for competency in competency_list:
                    st.write(f"##### {competency['competency']}")
                    if competency['description']:
                        st.write("Deskripsi:")
                        with st.container(border=True):
                            st.write(f"{competency['description']}")
                    else:
                        st.error('Error: Deskripsi kompetensi tidak ditemukan.', icon="🚨")
                    
                    if competency["levels"]:
                        st.write("Level:")
                        with st.container(border=True):
                            for level in competency["levels"]:
                                st.write(f"{level['level_name']}: {level['level_description']}")
                    else:
                        st.info('Info: Deskripsi level kompetensi tidak ditemukan.', icon="ℹ️")
            else:
                st.write(f"**Kompetensi tidak ditemukan.**")

    selected_level = df_pito_level[df_pito_level['id_level_set'] == selected_option_level_set]
    with st.container(border=True):
        #Level yang dipilih
        if not selected_option_level_set:
            st.warning("Silahkan pilih set level")
        else:
            st.write(f'#### Set Level dari {selected_option_level_set}')
            if not selected_level.empty:
                st.write(f"Terdiri dari:")
                with st.container(border=True):
                    for index, row in selected_level.iterrows():
                        st.write(f"**{row['value_level']}**. {row['NAMA LEVEL']}")
            else:
                st.error(f"Level set tidak ditemukan.", icon="🚨")

    #button untuk upload audio dan untuk mengupload audio menggunakan tipe file mp3, m4a dan wav
    st.markdown("Upload File Audio Anda")
    audio_file = st.file_uploader("Pilih File Audio", type=["mp3", "m4a", "wav",])

    id_level_set_fix, nama_level = get_level_set_from_audio_table(id_input_id_kandidat)

    df_pito_level['id_level_set'] = df_pito_level['id_level_set'].astype(str)
    df_pito_level['NAMA LEVEL'] = df_pito_level['NAMA LEVEL'].astype(str)

    if not id_level_set_fix:
        id_level_set_fix = selected_option_level_set
        nama_level = None

    id_level_set_fix = str(id_level_set_fix)
    filtered_levels_predict_competency = df_pito_level[df_pito_level['id_level_set'] == id_level_set_fix]
    level_names = filtered_levels_predict_competency['NAMA LEVEL'].tolist()
    #st.write(f"Level names: {level_names}")  #debug
    #st.write(f"Filtered levels predict competency: {filtered_levels_predict_competency}")  #debug
    dropdown_options_predict_competency = filtered_levels_predict_competency['NAMA LEVEL'].tolist()
    #st.write(f"Dropdown options predict competency: {dropdown_options_predict_competency}") #debug
        
    if st.button("Upload, Transcribe, dan Prediksi"):
        if audio_file is not None:

            st.write(f"Selected option level set: {selected_option_level_set}") #debug

            file_name = audio_file.name
            audio_bytes = audio_file.getvalue()
            audio_file_copy = io.BytesIO(audio_bytes)

            s3_url = upload_audio_to_s3(audio_file_copy, file_name)
            if not s3_url:
                st.error("Gagal mengunggah ke S3.")
                st.stop()

            st.success(f"Step 1/5: File {file_name} berhasil diunggah ke S3.")

            tz = pytz.timezone('Asia/Jakarta')
            conn = create_db_connection()
            cursor = conn.cursor()
            selected_id_product = int(selected_product['id_product'].iloc[0])
            selected_option_num_speaker = int(selected_option_num_speaker)
            insert_query = """
            INSERT INTO txtan_audio (registration_id, date, num_speakers, id_product, id_level_set, kode_assessor, audio_file_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            data = (id_input_id_kandidat, datetime.now(tz), selected_option_num_speaker, selected_id_product, selected_option_level_set, id_input_kode_assessor, file_name)
            cursor.execute(insert_query, data)
            conn.commit()
            id_audio = cursor.lastrowid
            cursor.close()
            conn.close()

            st.success("Step 2/5: Metadata audio disimpan di database.")
            
            file_extension = file_name.split('.')[-1].lower()
            audio_file_copy = io.BytesIO(audio_bytes)  
            audio_file_copy.seek(0)  

            transcript = transcribe_audio_gemini(audio_file_copy, file_extension, id_input_id_kandidat)
            st.write(f"Isi transkrip: {transcript}") #debug

            if transcript:
                if "transkrip" not in transcript or "transkrip" not in transcript["transkrip"] or not isinstance(transcript["transkrip"]["transkrip"], list):
                    st.error("Format JSON tidak sesuai: 'transkrip' tidak ditemukan atau bukan list.")
                else:
                    for entry in transcript["transkrip"]["transkrip"]:
                        if not isinstance(entry, dict) or "text" not in entry or "speaker" not in entry:
                            st.error(f"Format entry salah: {entry}")
                        else:
                            insert_into_separator(id_audio, id_input_id_kandidat, entry["text"], entry["speaker"], 0, 0)

                    st.success("Step 3/5: Transkripsi disimpan.")

                    # Jalankan prediksi kompetensi
                    st.write(f"Level untuk predictor: {dropdown_options_predict_competency}")
                    predictor(id_input_id_kandidat, dropdown_options_predict_competency)

            else:
                st.error("Transkripsi gagal dilakukan oleh Gemini API.")

#=======================================================================================================
#=============================================TAB 2=====================================================
#=======================================================================================================
with tab2:
    with st.container():
        if id_input_id_kandidat:
            df_transkrip = get_transkrip_data(id_input_id_kandidat)
            df_transkrip_reset = df_transkrip.reset_index(drop=True)
            table_html = df_transkrip_reset.to_html(index=False, escape=False)
            st.markdown("""
                <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    text-align: left;
                    vertical-align: top;
                    padding: 8px;
                    border: 1px solid #ddd;
                    word-wrap: break-word;
                    white-space: pre-wrap;
                }
                th {
                    background-color: #00;
                }
                </style>
            """, unsafe_allow_html=True)
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.warning("ID Kandidat tidak ditemukan/kosong")

#=======================================================================================================
#=============================================TAB 3=====================================================
#=======================================================================================================
with tab3:
    with st.container(border=True):
        st.write("Pilihan 'kosong' ada bisa dipilih jika dirasa memang tidak muncul di Assessor")
        st.write("Dropdown kompetensi dan level kompetensi **di sidebar** tidak akan mengubah pilihan level di bagian ini")

    with st.container():
        if id_input_id_kandidat:
            df_result_prediction = get_result_data(id_input_id_kandidat)

            if 'original_results' not in st.session_state:
                st.session_state['original_results'] = df_result_prediction.copy()

            df_result_prediction = st.session_state['original_results']

            id_level_set_fix, nama_level = get_level_set_from_audio_table(id_input_id_kandidat)
            #st.write(f"Nilai dari get_level_set_from_audio_table: id_level_set_fix={id_level_set_fix}, nama_level={nama_level}") #debug

            filtered_levels = df_pito_level[df_pito_level['id_level_set'] == id_level_set_fix]
            
            dropdown_options = filtered_levels['NAMA LEVEL'].tolist()
            dropdown_options.insert(0, '')

            so_values = get_all_so_values(id_input_id_kandidat)
            so_dict = {comp[0]: (comp[1], comp[2]) for comp in so_values}

            # Initialize a dictionary to track changes for save
            changes_to_save = []

            for i, row in enumerate(df_result_prediction.itertuples()):
                st.markdown(f"##### {row.competency}")
                st.write(f"###### Level: {row.level}")
                st.write(f"###### Alasan muncul: {row.reason}")

                so_level_key = f"dropdown_{i}"
                so_reason_key = f"text_input_{i}"

                current_so_level_value, current_so_reason_value = so_dict.get(row.competency, ("", ""))

                if f"prev_so_level_{i}" not in st.session_state:
                    st.session_state[f"prev_so_level_{i}"] = current_so_level_value
                if f"prev_so_reason_{i}" not in st.session_state:
                    st.session_state[f"prev_so_reason_{i}"] = current_so_reason_value

                # Debug Output: Checking the values before comparison
                # st.write(f"DEBUG - Row {i} Competency: {row.competency}")
                # st.write(f"DEBUG - Previous Level: {st.session_state[f'prev_so_level_{i}']}, Current Level: {current_so_level_value}")
                # st.write(f"DEBUG - Previous Reason: {st.session_state[f'prev_so_reason_{i}']}, Current Reason: {current_so_reason_value}")

                so_level = st.selectbox(
                    f"SO Level {row.competency}", 
                    dropdown_options, 
                    key=so_level_key,
                    index=dropdown_options.index(current_so_level_value) if current_so_level_value in dropdown_options else 0
                )

                so_reason = st.text_area(
                    f"Keterangan (opsional)", 
                    value=current_so_reason_value if current_so_reason_value else "",
                    key=f"so_reason_{row.competency}_{i}"
                )

                # Debug Output: Checking the values after input
                # st.write(f"DEBUG - After Input - Level: {so_level}, Reason: {so_reason}")

                # Track changes explicitly by comparing values
                if (so_level != current_so_level_value) or (so_reason != current_so_reason_value):
                    changes_to_save.append((row.competency, row.level, row.reason, so_level, so_reason, id_input_id_kandidat))
                    st.session_state[f"prev_so_level_{i}"] = so_level
                    st.session_state[f"prev_so_reason_{i}"] = so_reason
                    # st.write(f"DEBUG - Change detected for {row.competency}: {so_level} / {so_reason}")  # Debug output

            # Add a "Save" button at the end
            if st.button("Save Changes"):
                if changes_to_save:
                    try:
                        # Connect to the DB and update the records
                        conn = create_db_connection()
                        for change in changes_to_save:
                            competency, level, reason, so_level, so_reason, registration_id = change
                            update_single_entry_db(conn, competency, level, reason, so_level, so_reason, registration_id)
                        st.success("Perubahan berhasil disimpan!")
                    except Exception as e:
                        st.error(f"Error saving changes: {e}")
                else:
                    st.warning("Perubahan yang Anda lakukan sama dengan yang sudah disimpan")
        else:
            st.warning("ID Kandidat tidak ditemukan/kosong")

#=======================================================================================================
#=============================================TAB 1=====================================================
#=======================================================================================================
with tab4:
    with st.container(border=True):
        st.write("Berikut adalah fitur dimana Anda bisa menambahkan parameter ke sistem")

    subtab1, subtab2, subtab3 = st.tabs(["⚙️ <admin> Input Produk", "⚙️ <admin> Input Level", "⚙️ <admin> Input Assessor"])

    #=============================================TAB 1=====================================================
    with subtab1:
        if 'competencies' not in st.session_state:
            st.session_state['competencies'] = []

        with st.form(key='input_form'):
            input_name_product = st.text_input('Name Product', key='name_product')

            temp_competency = st.text_input('Competency', key='input_competency_temp')
            temp_description = st.text_area('Description', key='input_description_temp')

            level_value = st.number_input('Level Value', step=1, key='level_value')
            level_name = st.text_input('Level Name', key='level_name')
            level_description = st.text_area('Level Description', key='level_description')

            if st.form_submit_button(label='Add Competency Level'):
                if level_name and level_description:
                    if 'competency_levels' not in st.session_state:
                        st.session_state['competency_levels'] = []
                    st.session_state['competency_levels'].append({
                        "value": level_value,
                        "name": level_name,
                        "description": level_description,
                    })
                    st.success(f"Level kompetensi '{level_name}' ditambahkan.")

            if st.form_submit_button(label='Add Competency'):
                if temp_competency and temp_description:
                    if 'competencies' not in st.session_state:
                        st.session_state['competencies'] = []
                    st.session_state['competencies'].append((temp_competency, temp_description, st.session_state.get('competency_levels', [])))
                    st.success(f"Competency '{temp_competency}' ditambahkan.")
                    st.session_state['competency_levels'] = []
            
            st.write("Competencies yang sudah ditambahkan:")
            for idx, (competency, description, levels) in enumerate(st.session_state['competencies']):
                st.write(f"{idx + 1}. Competency: {competency}, Description: {description}")
                if levels:
                    for level in levels:
                        st.write(f"    - Level Value: {level['value']}, Level Name: {level['name']}, Level Description: {level['description']}")

            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            if input_name_product:  
                if is_product_exists(input_name_product):
                    st.error(f"Nama produk '{input_name_product}' sudah ada. Mohon gunakan nama lain.")
                else:
                    try:
                        conn = create_db_connection()
                        cursor = conn.cursor()
                        
                        query_product = """
                            INSERT INTO pito_product (name_product) 
                            VALUES (%s)
                        """
                        cursor.execute(query_product, (input_name_product,))
                        conn.commit()  

                        id_product = cursor.lastrowid

                        save_competencies_to_db(id_product)

                        st.success("Data produk, kompetensi, dan level kompetensi berhasil dimasukkan!")
                    except Exception as e:
                        st.error(f"Error saat menyimpan data: {e}")
                    finally:
                        cursor.close()
                        conn.close()
            else:
                st.error("Mohon masukkan nama produk sebelum menyimpan.")
                
    #=============================================SUBTAB 2=====================================================
    with subtab2:
        if 'new_levels_name' not in st.session_state:
            st.session_state['new_levels_name'] = []
        if 'new_levels_value' not in st.session_state:
            st.session_state['new_levels_value'] = []

        with st.container(border=True):
            level_set_name = st.text_input("Nama Level Set Baru", key="tab5_level_set")

            if level_set_name:
                existing_levels = get_existing_levels(level_set_name)
                if existing_levels:
                    st.warning(f"Set level '{level_set_name}' sudah ada, menampilkan level yang sudah ada.")
                    if not st.session_state['new_levels_name']: 
                        for name, value in existing_levels:
                            st.session_state['new_levels_name'].append(name)
                            st.session_state['new_levels_value'].append(value)

            input_level_name = st.text_input("Nama Level", key="tab5_nama_level")
            input_level_number = st.number_input("Masukkan Value Level", key="tab5_value_level", step=1)

            if st.button("Add Level", key="button_add_level"):
                if input_level_name and input_level_number:
                    if input_level_name in st.session_state['new_levels_name']:
                        st.error(f"Level dengan nama '{input_level_name}' sudah ada.")
                    else:
                        st.session_state['new_levels_name'].append(input_level_name)
                        st.session_state['new_levels_value'].append(input_level_number)
                        st.success(f"Level {input_level_name} dengan value {input_level_number} ditambahkan.")
                else:
                    st.error("Mohon masukkan nama level dan value level sebelum menambahkannya.")

            if st.session_state['new_levels_name']:
                st.write("Level yang sudah ditambahkan:")
                for i, (name, value) in enumerate(zip(st.session_state['new_levels_name'], st.session_state['new_levels_value'])):
                    st.write(f"{i+1}. Nama Level: {name}, Value Level: {value}")

                    if st.button(f"Hapus Level {name}", key=f"delete_{i}"):
                        st.session_state['new_levels_name'].pop(i)
                        st.session_state['new_levels_value'].pop(i)
                        st.success(f"Level '{name}' berhasil dihapus.")
                        st.experimental_rerun() 
            
            if st.button("Simpan Set Kompetensi", key="save_level"):
                if level_set_name and st.session_state['new_levels_name']:
                    
                    save_level_set_to_db(level_set_name, st.session_state['new_levels_name'], st.session_state['new_levels_value'])
                    
                    st.session_state['new_levels_name'] = []
                    st.session_state['new_levels_value'] = []
                    st.success("Set level berhasil ditambahkan!")
                else:
                    st.error("Mohon masukkan nama set level dan setidaknya satu level sebelum menyimpan.")

    #=============================================SUBTAB 3=====================================================
    with subtab3:
        st.write('Input Assessor')

        input_assessor_code = st.text_input("Kode Assessor (Huruf Kapital)")
        input_assessor_name = st.text_input("Nama Assessor")

        if st.button("Simpan Assessor"):
            if input_assessor_code and input_assessor_name:
                save_assessor_to_db(input_assessor_code, input_assessor_name)
            else:
                st.error("Mohon masukkan kode dan nama assessor.")
