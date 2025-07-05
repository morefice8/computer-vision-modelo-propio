from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}


    def get_clustering_model(self, image, n_clusters=2):
        image_2d = image.reshape((-1, 3)) #Aplano la imagen a 2D    
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1) #Aplico el modelo de clustering KMeans
        kmeans.fit(image_2d) #Entreno el modelo de clustering KMeans
        return kmeans


    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] #Corte la imagen del jugador usando las coordenadas del bounding box
        top_half_image = image[0:int(image.shape[0]/2),:] #Corte la mitad superior de la imagen
        kmeans = self.get_clustering_model(top_half_image, n_clusters=2) #Aplico el modelo de clustering KMeans a la mitad superior de la imagen
        labels = kmeans.labels_ #Obtengo las etiquetas de los clusters
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1]) #Reformo la imagen a su forma original
        corner_cluster = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]  # Esquinas de la imagen agrupada
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)  # No esquinas de la imagen agrupada
        player_cluster = 1 - non_player_cluster  # Cluster del jugador
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1) #Aplico el modelo de clustering KMeans
        kmeans.fit(player_colors)

        self.kmeans = kmeans #Guardo el modelo de clustering KMeans

        self.team_colors[1] = kmeans.cluster_centers_[0] #Asigno el color del equipo 1
        self.team_colors[2] = kmeans.cluster_centers_[1] #Asigno el color del equipo 2
        
        
    def get_player_team(self, frame, player_bbox, player_id):
        # Asigna el equipo según el color del jugador
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        player_color = self.get_player_color(frame, player_bbox) #Obtengo el color del jugador
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0] #Asigno el equipo según el color del jugador
        team_id += 1
        self.player_team_dict[player_id] = team_id
        return team_id