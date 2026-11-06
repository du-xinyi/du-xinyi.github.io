#!/bin/bash
echo '#!/bin/bash' > chrome_cn.sh
echo 'LANGUAGE=zh_CN /usr/bin/google-chrome-stable "$@"' >> chrome_cn.sh
sudo mv chrome_cn.sh /usr/bin/chrome_cn.sh
sudo chmod +x /usr/bin/chrome_cn.sh
mkdir -p ~/.local/share/applications
cp /usr/share/applications/google-chrome.desktop ~/.local/share/applications/
sed -i "s@/usr/bin/google-chrome-stable@/usr/bin/chrome_cn.sh@g" ~/.local/share/applications/google-chrome.desktop
echo 'Chinesization Complete'
