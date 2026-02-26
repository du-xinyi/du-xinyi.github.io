#!/bin/bash
touch chrome_cn.sh
echo '#!/bin/bash' > chrome_cn.sh
echo 'LANGUAGE=zh_cn /usr/bin/google-chrome-stable $@' >> chrome_cn.sh
sudo chmod a+x chrome_cn.sh
sudo mv chrome_cn.sh /usr/bin/chrome_cn.sh
sudo sed -i "s@/usr/bin/google-chrome-stable@/usr/bin/chrome_cn.sh@g" /usr/share/applications/google-chrome.desktop
echo 'Chinesization Complete'