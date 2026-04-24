import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os


# CONFIGURACIÓN DE INTERFAZ-------------------------------------------------------------------------------------------------------

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg_main": "#FDF2F8",      # Rosa muy claro
    "bg_frame": "#F0F8FF",     # Azul muy claro
    "btn_fg": "#FFB6C1",       # Rosa claro
    "btn_hover": "#87CEFA",    # Azul claro
    "text_dark": "#1E3A8A",    # Azul oscuro (títulos)
    "text_body": "#374151",    # Gris azulado (contenido)
    "res_bg": "#FFFFFF"        # Blanco para resultados
}

CSV_PATH = "datos_redes.csv"
CREDS_PATH = "credenciales.txt"

def load_data():
    if not os.path.exists(CSV_PATH):
        messagebox.showerror("Error", f"No se encontró '{CSV_PATH}'.")
        return None
    return pd.read_csv(CSV_PATH)

def load_credentials():
    if not os.path.exists(CREDS_PATH):
        with open(CREDS_PATH, "w", encoding="utf-8") as f:
            f.write("admin,1234\nuser,pass123")
        messagebox.showinfo("Info", "Se creó 'credenciales.txt' con usuarios de prueba.")
    creds = {}
    with open(CREDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and "," in line:
                u, p = line.strip().split(",", 1)
                creds[u] = p
    return creds


# VENTANA DE LOGIN------------------------------------------------------------------------------------------------------------------------------------

def show_login_window():
    win = ctk.CTk()
    win.title("Acceso")
    win.geometry("300x220")
    win.resizable(False, False)
    win.configure(fg_color=COLORS["bg_main"])

    ctk.CTkLabel(win, text="Inicio de Sesión", font=("Arial", 16, "bold"), text_color=COLORS["text_dark"]).pack(pady=15)
    
    ent_user = ctk.CTkEntry(win, placeholder_text="Usuario", width=220)
    ent_user.pack(pady=5)
    ent_pass = ctk.CTkEntry(win, placeholder_text="Contraseña", width=220, show="*")
    ent_pass.pack(pady=5)

    def validate():
        u, p = ent_user.get().strip(), ent_pass.get().strip()
        if not u or not p:
            messagebox.showwarning("Validación", "Completa ambos campos.")
            return
        creds = load_credentials()
        if creds.get(u) == p:
            win.destroy()
            start_main_app()
        else:
            messagebox.showerror("Acceso Denegado", "Credenciales incorrectas.")

    ctk.CTkButton(win, text="Entrar", command=validate, width=180,
                  fg_color=COLORS["btn_fg"], hover_color=COLORS["btn_hover"],
                  text_color=COLORS["text_dark"], font=("Arial", 12, "bold")).pack(pady=15)
    win.mainloop()

# CONSULTAS --------------------------------------------------------------------------------------------------------------------

def run_query(q_id, df, results_frame):
    for w in results_frame.winfo_children():
        w.destroy()

    text_res = ""

    try:
        # 1. Ansiedad en Mujeres (Instagram >3h)
        if q_id == 1:
            mask = (df["gender"] == "female") & (df["platform_usage"] == "Instagram") & (df["daily_social_media_hours"] > 3)
            avg = df.loc[mask, "anxiety_level"].mean()
            count = mask.sum()
            text_res = f"CONSULTA 1: ANSIEDAD EN MUJERES (INSTAGRAM >3H/DÍA)\n"
            text_res += f"Promedio nivel_ansiedad: {avg:.2f} / 10\n"
            text_res += f"Registros evaluados: {count}\n"
            text_res += f"Columnas: gender, platform_usage, daily_social_media_hours, anxiety_level\n\n"
            text_res += "INTERPRETACIÓN:\n"
            text_res += "Este valor indica el nivel promedio de ansiedad entre mujeres que usan Instagram más de 3 horas diarias. "
            text_res += "Un puntaje cercano a 5-10 sugiere que el uso prolongado podría estar asociado con mayor estrés emocional "
            text_res += "en este grupo. Un valor bajo (<4) indicaría que, en esta muestra, el tiempo de uso no eleva significativamente la ansiedad."

        # 2. Correlación TikTok vs Sueño
        elif q_id == 2:
            sub = df[df["platform_usage"] == "TikTok"].dropna(subset=["daily_social_media_hours", "sleep_hours"])
            corr = sub[["daily_social_media_hours", "sleep_hours"]].corr().iloc[0, 1]
            text_res = f"CONSULTA 2: CORRELACIÓN TIKTOK vs SUEÑO\n"
            text_res += f"Coeficiente de correlación (Pearson): {corr:.4f}\n"
            text_res += f"Columnas: platform_usage, daily_social_media_hours, sleep_hours\n\n"
            text_res += "INTERPRETACIÓN:\n"
            text_res += f"El coeficiente es {corr:.2f}. "
            text_res += "Si el valor es negativo, indica que a mayor uso de TikTok, menores horas de sueño. "
            text_res += "Si es positivo, sugiere lo contrario (aunque biológicamente es poco común). "
            text_res += "Valores cercanos a ±0.5 o ±1 indican una relación fuerte; cercanos a 0 sugieren que el tiempo en TikTok "
            text_res += "no determina directamente las horas de descanso en esta muestra."

        # 3. Rendimiento y Estrés por Red (14-16 años)
        elif q_id == 3:
            mask = df["age"].between(14, 16)
            stats = df[mask].groupby("platform_usage").agg(
                Rendimiento=("academic_performance", "mean"),
                Estrés=("stress_level", "mean")
            ).sort_values("Rendimiento", ascending=False).round(2)
            text_res = f"CONSULTA 3: RENDIMIENTO Y ESTRÉS POR RED (14-16 AÑOS)\n"
            text_res += f"{stats.to_string()}\n"
            text_res += f"Columnas: age, platform_usage, academic_performance, stress_level\n\n"
            text_res += "INTERPRETACIÓN:\n"
            text_res += "Muestra qué plataformas se asocian con mejor rendimiento académico y menores niveles de estrés en adolescentes. "
            text_res += "Las redes en la parte superior suelen tener un impacto más equilibrado o positivo en el desarrollo cognitivo y emocional "
            text_res += "de este grupo con la misma edad, mientras que las de la parte inferior podrían estar vinculadas a mayor distracción o presión social."

        # 4. Prevalencia Depresión (Pantalla >2h)
        elif q_id == 4:
            mask = df["screen_time_before_sleep"] > 2.0
            dep_rate = df[mask].groupby("social_interaction_level")["depression_label"].mean() * 100
            text_res = f"CONSULTA 4: PREVALENCIA DE DEPRESIÓN (PANTALLA >2H ANTES DE DORMIR)\n"
            text_res += f"Por nivel de interacción social:\n"
            text_res += dep_rate.round(2).astype(str).add("%").to_string() + "\n"
            text_res += f"Columnas: screen_time_before_sleep, depression_label, social_interaction_level\n\n"
            text_res += "INTERPRETACIÓN:\n"
            text_res += "Revela qué porcentaje de usuarios presentan síntomas depresivos tras usar la pantalla >2h antes de dormir. "
            text_res += "Si el porcentaje es más alto en el nivel 'low', sugiere que el aislamiento social nocturno agrava el riesgo. "
            text_res += "Un % similar entre todos los niveles indicaría que la luz azul y la sobrestimulación afectan por igual, "
            text_res += "independientemente del contexto social del usuario."

        # 5. Actividad Física por Género y Plataforma
        elif q_id == 5:
            pivot = df.groupby(["gender", "platform_usage"])["physical_activity"].mean().unstack().round(2)
            text_res = f"CONSULTA 5: ACTIVIDAD FÍSICA PROMEDIO (GÉNERO x PLATAFORMA)\n"
            text_res += f"{pivot.to_string()}\n"
            text_res += f"Columnas: physical_activity, gender, platform_usage\n\n"
            text_res += "INTERPRETACIÓN:\n"
            text_res += "Compara cuánta actividad física realizan hombres y mujeres según la red que más usan. "
            text_res += "Valores por debajo de 1.0 apuntan a un estilo de vida sedentario, lo que podría correlacionarse con "
            text_res += "mayor tiempo sentado frente a dispositivos. Diferencias marcadas entre plataformas ayudan a identificar "
            text_res += "cuáles fomentan más movimiento o, por el contrario, incentivan el sedentarismo."

        # 6. Regresión Adicción vs Ansiedad
        elif q_id == 6:
            x, y = df["addiction_level"].values, df["anxiety_level"].values
            slope, intercept = np.polyfit(x, y, 1)
            text_res = f"CONSULTA 6: REGRESIÓN LINEAL ADICCIÓN vs ANSIEDAD\n"
            text_res += f"Ecuación: y = {slope:.3f}x + {intercept:.3f}\n"
            text_res += f"Columnas: addiction_level, anxiety_level\n\n"
            text_res += "INTERPRETACIÓN:\n"
            text_res += f"La pendiente ({slope:.3f}) indica cuánto aumenta la ansiedad por cada punto adicional en la escala de adicción. "
            text_res += "Una pendiente positiva confirma que, estadísticamente, mayor dependencia a las redes coincide con mayor ansiedad. "
            text_res += "Si la pendiente fuera 0 o negativa, no habría evidencia de que la adicción impacte directamente el nivel de ansiedad."

        # 7. Uso Extremo (>6h) por Edad y Género
        elif q_id == 7:
            ext = df[df["daily_social_media_hours"] > 6.0]
            dist = ext.groupby([pd.cut(ext["age"], bins=[0, 16, 20, 100], labels=["≤16", "17-20", "≥21"]), "gender"]).size().unstack(fill_value=0)
            text_res = f"CONSULTA 7: DISTRIBUCIÓN USO EXTREMO (>6H/DÍA)\n"
            text_res += f"{dist.to_string()}\n"
            text_res += f"Columnas: daily_social_media_hours, age, gender\n\n"
            text_res += "INTERPRETACIÓN:\n"
            text_res += "Identifica qué franjas de edad y género concentran más usuarios con consumo extremo. "
            text_res += "Estos datos son útiles para dirigir campañas de salud digital. Un número alto en '≤16' señala "
            text_res += "vulnerabilidad temprana a la sobreexposición digital, mientras que concentraciones en adultos jóvenes "
            text_res += "podrían relacionarse con hábitos laborales o de ocio nocturno."

        # 8. Matriz de Correlación (Matplotlib)
        elif q_id == 8:
            cols = ["daily_social_media_hours", "sleep_hours", "screen_time_before_sleep", 
                    "academic_performance", "physical_activity", "stress_level", "anxiety_level", "addiction_level"]
            corr = df[cols].corr().round(2)
            text_res = f"CONSULTA 8: MATRIZ DE CORRELACIÓN \n"
            text_res += f"{corr.to_string()}\n"
            text_res += f"Columnas: Todas las numéricas\n\n"
            text_res += "INTERPRETACIÓN:\n"
            text_res += "Cada celda muestra la relación entre dos variables. 1.00 = relación perfecta positiva, "
            text_res += "-1.00 = relación perfecta negativa, 0.00 = sin relación. Busca valores >|0.5| para identificar "
            text_res += "vínculos estadísticamente relevantes. Por ejemplo, si 'addiction_level' y 'anxiety_level' tienen "
            text_res += "0.6+, confirma que a mayor adicción, mayor ansiedad. Si 'sleep_hours' y 'stress_level' tienen -0.4, "
            text_res += "indica que dormir más se asocia con menos estrés."

        # 9. Impacto de la Interacción Social
        elif q_id == 9:
            stats = df.groupby("social_interaction_level").agg(
                Estrés=("stress_level", "mean"),
                Rendimiento=("academic_performance", "mean"),
                Ansiedad=("anxiety_level", "mean")
            ).round(2)
            text_res = f"CONSULTA 9: IMPACTO DE LA INTERACCIÓN SOCIAL\n"
            text_res += f"{stats.to_string()}\n"
            text_res += f"Columnas: social_interaction_level, stress_level, academic_performance, anxiety_level\n\n"
            text_res += "INTERPRETACIÓN:\n"
            text_res += "Analiza cómo el nivel de interacción social modera el bienestar psicológico y académico. "
            text_res += "Si el grupo 'high' muestra menor estrés y mayor rendimiento, sugiere que el apoyo social actúa como "
            text_res += "factor protector frente al impacto negativo de las redes. Diferencias notables entre niveles "
            text_res += "indican que el contexto relacional es clave para entender los efectos del uso digital."

        # 10. Perfil de Riesgo Multivariable
        elif q_id == 10:
            risk = df[(df["addiction_level"] >= 7) & (df["physical_activity"] <= 0.5)]
            profile = risk.groupby(["platform_usage", "gender"]).agg(
                Edad_Promedio=("age", "mean"),
                Sueño_Promedio=("sleep_hours", "mean"),
                Total_Usuarios=("age", "count")
            ).round(2)
            text_res = f"CONSULTA 10: PERFIL DE RIESGO (ADICCIÓN ≥7, ACTIVIDAD ≤0.5)\n"
            if profile.empty:
                text_res += "No se encontraron registros con este perfil extremo en la muestra.\n"
            else:
                text_res += f"{profile.to_string()}\n"
            text_res += f"Columnas: addiction_level, physical_activity, platform_usage, gender, age, sleep_hours\n\n"
            text_res += "INTERPRETACIÓN:\n"
            text_res += "Aísla un grupo de alta vulnerabilidad: usuarios con dependencia severa y mínima actividad física. "
            text_res += "La edad promedio y las horas de sueño revelan si este grupo sufre privación de descanso o si pertenece "
            text_res += "a una etapa crítica del desarrollo. Estos datos son esenciales para diseñar intervenciones clínicas "
            text_res += "o programas de desconexión digital dirigidos a los casos más sensibles."

    except Exception as e:
        text_res = f"Error al procesar la consulta:\n{str(e)}"

    # Renderizar en el frame
    txt = ctk.CTkTextbox(results_frame, font=("Consolas", 12), fg_color=COLORS["res_bg"], text_color=COLORS["text_body"])
    txt.pack(fill="both", expand=True, padx=15, pady=15)
    txt.insert("0.0", text_res)
    txt.configure(state="disabled")


# INTERFAZ PRINCIPAL---------------------------------------------------------------------------------------------------------------

class DashboardApp(ctk.CTk):
    def __init__(self, df):
        super().__init__()
        self.title("Análisis de Redes Sociales")
        self.geometry("1050x750")
        self.df = df
        self.configure(fg_color=COLORS["bg_main"])
        
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        menu = ctk.CTkFrame(self, corner_radius=15, fg_color=COLORS["bg_frame"])
        menu.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(menu, text="MENÚ DE CONSULTAS", font=("Arial", 16, "bold"), text_color=COLORS["text_dark"]).pack(pady=(15, 10))

        scroll = ctk.CTkScrollableFrame(menu, width=230, height=620, fg_color=COLORS["bg_frame"])
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        queries = [
            ("1. Ansiedad (Mujeres, IG >3h)", 1),
            ("2. Correlación TikTok vs Sueño", 2),
            ("3. Rendimiento/Estrés (14-16a)", 3),
            ("4. Depresión x Pantalla Nocturna", 4),
            ("5. Act. Física x Género/Red", 5),
            ("6. Adicción vs Ansiedad (Numpy)", 6),
            ("7. Uso Extremo (>6h) x Edad/Gén", 7),
            ("8. Matriz Correlación", 8),
            ("9. Impacto Interacción Social", 9),
            ("10. Perfil de Riesgo Multivar.", 10)
        ]

        for label, q_id in queries:
            btn = ctk.CTkButton(scroll, text=label, corner_radius=8,
                                fg_color=COLORS["btn_fg"], hover_color=COLORS["btn_hover"],
                                text_color=COLORS["text_dark"], font=("Arial", 11, "bold"),
                                command=lambda x=q_id: run_query(x, self.df, self.res_frame))
            btn.pack(fill="x", padx=8, pady=6)

        self.res_frame = ctk.CTkFrame(self, corner_radius=15, fg_color=COLORS["bg_frame"])
        self.res_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.res_frame, text="📈 RESULTADOS", font=("Arial", 16, "bold"), text_color=COLORS["text_dark"]).pack(pady=10)
        ctk.CTkLabel(self.res_frame, text="Selecciona una consulta para visualizar...", 
                     font=("Arial", 12), text_color="gray").pack(pady=40)

def start_main_app():
    df = load_data()
    if df is not None:
        DashboardApp(df).mainloop()

if __name__ == "__main__":
    show_login_window()