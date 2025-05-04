#!/bin/bash
# ETH-XGaze modelini doğru konuma yerleştirmek için yardımcı script

# Script'in bulunduğu konumu al
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
MODELS_DIR="$PROJECT_ROOT/models"

# Model klasörünün var olduğunu kontrol et, yoksa oluştur
if [ ! -d "$MODELS_DIR" ]; then
    echo "Modeller klasörü oluşturuluyor: $MODELS_DIR"
    mkdir -p "$MODELS_DIR"
fi

# Komut satırı argümanları
if [ $# -eq 0 ]; then
    echo "Kullanım: $0 <eth_xgaze_model_dosyası>"
    echo "Örnek: $0 ~/Downloads/eth_xgaze.onnx"
    exit 1
fi

MODEL_PATH="$1"
MODEL_FILENAME=$(basename "$MODEL_PATH")
MODEL_EXT="${MODEL_FILENAME##*.}"

# Model uzantısını kontrol et
if [ "$MODEL_EXT" = "onnx" ]; then
    TARGET_PATH="$MODELS_DIR/eth_xgaze_model.onnx"
    echo "ONNX modeli kopyalanıyor: $MODEL_PATH -> $TARGET_PATH"
elif [ "$MODEL_EXT" = "pth" ] || [ "$MODEL_EXT" = "tar" ]; then
    TARGET_PATH="$MODELS_DIR/eth_xgaze_model.pth"
    echo "PyTorch modeli kopyalanıyor: $MODEL_PATH -> $TARGET_PATH"
else
    echo "Hata: Desteklenmeyen model formatı: $MODEL_EXT"
    echo "Desteklenen formatlar: onnx, pth, tar"
    exit 1
fi

# Modelin var olduğunu kontrol et
if [ ! -f "$MODEL_PATH" ]; then
    echo "Hata: Model dosyası bulunamadı: $MODEL_PATH"
    exit 1
fi

# Modeli kopyala
cp "$MODEL_PATH" "$TARGET_PATH"

# Başarı durumunu kontrol et
if [ $? -eq 0 ]; then
    echo "Model başarıyla kopyalandı: $TARGET_PATH"
    
    # ONNX modeli için onnxruntime kurulumunu kontrol et
    if [ "$MODEL_EXT" = "onnx" ]; then
        pip list | grep -q "onnxruntime"
        if [ $? -ne 0 ]; then
            echo "onnxruntime paketi kurulu değil."
            read -p "onnxruntime paketini kurmak ister misiniz? (e/h): " INSTALL
            if [[ "$INSTALL" = "e" || "$INSTALL" = "E" ]]; then
                pip install onnxruntime
            else
                echo "ONNX modeli kullanmak için onnxruntime paketini manuel olarak kurmanız gerekecek:"
                echo "pip install onnxruntime"
            fi
        else
            echo "onnxruntime paketi zaten kurulu."
        fi
    fi
    
    echo "Sistem hazır!"
else
    echo "Hata: Model kopyalanırken bir sorun oluştu."
    exit 1
fi 