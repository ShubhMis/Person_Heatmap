
    def __generate_heatmap(self, frame, detector_meta,track_ids):
        """
        Update a persistent heatmap based on leg region detection.
        Only includes regions from the last 10 seconds.
        """
        height, width = frame.shape[:2]
        current_time = time.time()
        min_movement_thresh = 5
        track_ids = np.array(track_ids)
        if self.accum_image is None:
            self.accum_image = np.zeros((height, width), dtype=np.uint8)

        new_masks = []

        for idx,box in enumerate(detector_meta[:, :4]):
            x0, y0, x1, y1 = map(int, box)

            cx = int((x1+x0) / 2)
            cy = int(y1)

            t_id = int(track_ids[idx])

            # Compare with previous center (if any)
            if t_id in self.prev_centers:
                prev_cx, prev_cy = self.prev_centers[t_id]
                move_distance = np.hypot(cx - prev_cx, cy - prev_cy)
                if 5 < move_distance < 20:
                    continue
                    
            self.prev_centers[t_id] = (cx, cy) 

            # Create a new small mask with intensity
            mask = np.zeros((height, width), dtype=np.uint8)
            cv.circle(mask, (cx, cy), radius=8, color=7, thickness=-1)

            new_masks.append((current_time, mask))

        # Add new masks to the buffer
        self.heatmap_buffer.extend(new_masks)

        # Remove expired masks (older than  seconds)
        self.heatmap_buffer = [
            (t, m) for (t, m) in self.heatmap_buffer if current_time - t <= self.heatmap_expire_time
        ]

        # Recompute accum_image from valid masks
        self.accum_image = np.zeros((height, width), dtype=np.uint8)
        for _, m in self.heatmap_buffer:
            self.accum_image = cv.add(self.accum_image, m)

        # Apply color map and overlay
        amplified = cv.convertScaleAbs(self.accum_image, alpha=1.5)
        heatmap_overlay = cv.applyColorMap(amplified, cv.COLORMAP_JET)
        frame[:] = cv.addWeighted(frame, 0.5, heatmap_overlay, 0.7, 0)



    def __draw_flow_pattern(self, frame, detector_meta, track_ids):
        current_time = time.time()
        track_ids = np.array(track_ids)

        # Store current positions with timestamps
        for idx, box in enumerate(detector_meta[:, :4]):
            if idx >= len(track_ids):
                continue

            t_id = track_ids[idx]
            x0, y0, x1, y1 = map(int, box)
            cx = int((x0 + x1) / 2)
            cy = int((y0 + y1) / 2)

            self.trails[t_id].append((cx, cy, current_time))

        # Prune old points per track
        for t_id in list(self.trails.keys()):
            self.trails[t_id] = deque([
                (x, y, t)
                for (x, y, t) in self.trails[t_id]
                if current_time - t <= self.flow_expire_time
            ])

        # Create mask for drawing trails
        glow_mask = np.zeros_like(frame)

        for trail in self.trails.values():
            if len(trail) < 2:
                continue

            # Smooth positions (ignore timestamps)
            trail_list = list(trail)
            smoothed = []
            window_size = 7
            for i in range(len(trail_list)):
                window = trail_list[max(0, i - window_size + 1):i + 1]
                if not window:
                    continue
                avg_x = int(np.mean([pt[0] for pt in window]))
                avg_y = int(np.mean([pt[1] for pt in window]))
                smoothed.append((avg_x, avg_y))

            for j in range(1, len(smoothed)):
                pt1 = smoothed[j - 1]
                pt2 = smoothed[j]
                cv.line(glow_mask, pt1, pt2, (255, 255, 200), thickness=6)

        blurred = cv.GaussianBlur(glow_mask, (15, 15), 0)
        frame[:] = cv.addWeighted(frame, 1.0, blurred, 0.6, 0)
