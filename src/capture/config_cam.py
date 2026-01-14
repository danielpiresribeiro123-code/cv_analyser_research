import cv2
import socket
from typing import Dict, Optional, List, Tuple
from onvif import ONVIFCamera


# ======================================================
#  LISTAR DISPOSITIVOS DE VÍDEO (USB / WEBCAMS / VIRTUAIS)
# ======================================================
def listar_dispositivos_video(max_test: int = 5) -> None:
    """
    Lista dispositivos de vídeo locais disponíveis (USB, webcams virtuais, etc).

    Args:
        max_test: número máximo de índices a testar (0..max_test-1)
    """
    print("\n<----------- DISPOSITIVOS DE VÍDEO DETETADOS ----------->")

    found_any = False
    for i in range(max_test):
        # CAP_DSHOW é recomendado no Windows para evitar bloqueios
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            print(f"ID {i} → ✔ Disponível")
            cap.release()
            found_any = True
        else:
            print(f"ID {i} → ✖ Não disponível")

    if not found_any:
        print("\n⚠ Nenhum dispositivo de vídeo encontrado.")

    print("<------------------------------------------------------>\n")


# ======================================================
#             SCAN POR CÂMARAS IP NA REDE
# ======================================================
def procurar_ip_cameras() -> List[Tuple[str, int]]:
    """
    Procura dispositivos na rede local que respondam em portas típicas
    de câmaras IP / RTSP / ONVIF.

    NOTA:
        Isto deteta dispositivos ativos, não garante que sejam câmaras.

    Returns:
        Lista de tuplos (ip, porta)
    """
    print("\n<----------- PROCURA DE CÂMARAS IP (SCAN) ----------->")
    intervalo = input("Intervalo (ex: 192.168.1.1-192.168.1.255) → ").strip()

    # ---------------- Validação do intervalo ----------------
    try:
        inicio, fim = intervalo.split("-")
        base_i = inicio.split(".")
        base_f = fim.split(".")

        # Garantir que é a mesma sub-rede
        if base_i[:3] != base_f[:3]:
            print("❌ O intervalo deve estar na mesma sub-rede.")
            return []

        base = ".".join(base_i[:3])
        start = int(base_i[3])
        end = int(base_f[3])
    except Exception:
        print("❌ Formato inválido.")
        return []

    portas = [80, 554, 8000, 8899]  # HTTP, RTSP, ONVIF
    encontrados: List[Tuple[str, int]] = []

    print(f"A procurar entre {base}.{start} e {base}.{end} ...\n")

    for i in range(start, end + 1):
        ip = f"{base}.{i}"

        for porta in portas:
            sock = socket.socket()
            sock.settimeout(0.2)
            try:
                sock.connect((ip, porta))
                sock.close()
                print(f"✔ Dispositivo encontrado em {ip}:{porta}")
                encontrados.append((ip, porta))
                break
            except Exception:
                sock.close()
                continue

    if not encontrados:
        print("\n❌ Nenhuma câmara IP encontrada no intervalo.")
    else:
        print("\nDispositivos encontrados:")
        for idx, (ip, porta) in enumerate(encontrados, 1):
            print(f"{idx} - {ip}:{porta}")

    print("<------------------------------------------------------>\n")
    return encontrados


# ======================================================
#          MENU PRINCIPAL DE CONFIGURAÇÃO DE CÂMARA
# ======================================================
def escolher_camera_inicial(
    max_tests: int = 5,
    auto: bool = False,
    tipo: int | str = "",
    usb_source: int = 0,
    ip_source: str = ""
) -> Optional[Dict]:
    """
    Seleciona a fonte de vídeo e devolve uma configuração normalizada.

    Returns:
        dict com a forma:
        {
            "source": int | str,   # argumento direto para cv2.VideoCapture
            "type": "usb" | "ip",
            "onvif": dict | None   # apenas informativo
        }

        ou None se falhar.
    """

    # ---------------- Menu interativo ----------------
    if not auto:
        print("\n<------------------ SELEÇÃO DE CÂMARA ------------------>")
        print("1 - Câmara USB / integrada")
        print("2 - Câmara IP (RTSP ou ONVIF)")
        print("3 - Listar dispositivos de vídeo (USB)")
        print("4 - Procurar câmaras IP (scan de rede)")
        print("--------------------------------------------------------")

    # Em modo automático forçamos tipo para string
    tipo = str(tipo) if auto else input("Escolhe a opção → ").strip()

    # ======================================================
    # OPÇÃO 3 – LISTAR USB
    # ======================================================
    if tipo == "3":
        listar_dispositivos_video(max_tests)
        return escolher_camera_inicial(max_tests)

    # ======================================================
    # OPÇÃO 4 – SCAN IP
    # ======================================================
    if tipo == "4":
        resultados = procurar_ip_cameras()
        if not resultados:
            return escolher_camera_inicial(max_tests)

        escolha = input("Seleciona índice do dispositivo (Enter para cancelar) → ").strip()
        if not escolha.isdigit():
            return escolher_camera_inicial(max_tests)

        idx = int(escolha) - 1
        if idx < 0 or idx >= len(resultados):
            print("❌ Índice inválido.")
            return escolher_camera_inicial(max_tests)

        ip, _ = resultados[idx]
        rtsp_url = f"rtsp://{ip}:554/"

        print(f"Câmara IP selecionada automaticamente: {rtsp_url}")
        return {
            "source": rtsp_url,
            "type": "ip",
            "onvif": None
        }

    # ======================================================
    # OPÇÃO 1 – USB
    # ======================================================
    if tipo == "1":
        entrada = input("ID da câmara (0, 1, 2...). Enter → ").strip()

        camera_source = usb_source if auto else (
            0 if entrada == "" else int(entrada)
        )

        print(f"Câmara USB selecionada → ID {camera_source}")
        return {
            "source": camera_source,
            "type": "usb",
            "onvif": None
        }


    # ======================================================
    # OPÇÃO 2 – IP MANUAL / ONVIF
    # ======================================================
    if tipo == "2":
        if auto:
            if not ip_source.strip():
                print("❌ Nenhum URL fornecido.")
                return None

            print(f"Câmara IP selecionada automaticamente: {ip_source}")
            return {
                "source": ip_source.strip(),
                "type": "ip",
                "onvif": None
            }

        print("\nConfiguração de câmara IP:")
        print("1 - Introduzir URL RTSP manualmente")
        print("2 - Usar ONVIF para obter RTSP automático")
        sub = input("Escolhe → ").strip()

        # ----------- RTSP manual -----------
        if sub == "1":
            rtsp = input("Insere URL RTSP completo → ").strip()
            if not rtsp:
                print("❌ Nenhum URL fornecido.")
                return None

            return {
                "source": rtsp,
                "type": "ip",
                "onvif": None
            }

        # ----------- ONVIF automático -----------
        if sub == "2":
            ip = input("IP (ex: 192.168.1.96) → ").strip()
            port = int(input("Porta ONVIF (default 8899) → ").strip() or 8899)
            user = input("Utilizador ONVIF → ").strip()
            pwd = input("Password ONVIF → ").strip()

            try:
                cam = ONVIFCamera(ip, port, user, pwd)
                media = cam.create_media_service()
                profiles = media.GetProfiles()
                token = profiles[0].token

                uri = media.GetStreamUri({
                    "StreamSetup": {
                        "Stream": "RTP-Unicast",
                        "Transport": {"Protocol": "RTSP"}
                    },
                    "ProfileToken": token
                })

                return {
                    "source": uri.Uri,
                    "type": "ip",
                    "onvif": {
                        "ip": ip,
                        "port": port,
                        "user": user,
                        "pwd": pwd
                    }
                }

            except Exception as e:
                print(f"Erro ONVIF: {e}")
                return None

    print("❌ Opção inválida.")
    return None
