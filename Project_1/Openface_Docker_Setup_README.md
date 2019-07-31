=================================================
Docker Setup :
=================================================
docker pull bamos/openface
docker run -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash
cd /root/openface

=================================================
=================================================
Docker Run with Images in Container
=================================================

docker run -v <image-directory-in-container>:<directory-in-docker> -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash
cd /root/openface

=================================================
