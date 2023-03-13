import math

def thief_and_cops(grid, orientations, fov):
    n_cops = len(orientations)
    n_rows, n_cols = len(grid), len(grid[0])
    cops = [[0, 0] for _ in range(n_cops)]

    # Get positions of thief and cops
    for row in range(n_rows):
        for col in range(n_cols):
            if grid[row][col] == 'T':
                theif = [row, col]
            elif grid[row][col] != 0:
                cop_id = grid[row][col]
                cops[cop_id-1][0] = row
                cops[cop_id-1][1] = col
    
    seen = [[0] * n_cols for _ in range(n_rows)]
    cops_visible = []

    # Check if two FoVs overlap
    def overlap(left_row, left_col, right_row, right_col):
        nonlocal n_rows, cop_x, cop_y, cop_left_r, cop_right_r
        left_x, left_y = xy_coordinate(left_row, left_col, n_rows)
        right_x, right_y = xy_coordinate(right_row, right_col, n_rows)
        left_r = math.atan2(left_y - cop_y, left_x - cop_x)
        right_r = math.atan2(right_y - cop_y, right_x - cop_x)

        if left_r < 0:
            left_r += 2*math.pi
        if right_r < 0:
            right_r += 2*math.pi

        if left_r < right_r and cop_left_r < cop_right_r:
            return True
        elif left_r < right_r:
            return max(right_r, cop_right_r) < min(left_r+2*math.pi, cop_left_r) or max(right_r, cop_right_r+2*math.pi) < min(left_r+2*math.pi, cop_left_r+2*math.pi)
        elif cop_left_r < cop_right_r:
            return max(right_r, cop_right_r) < min(left_r, cop_left_r+2*math.pi) or max(right_r+2*math.pi, cop_right_r) < min(left_r+2*math.pi, cop_left_r+2*math.pi)
        
        return max(right_r, cop_right_r) < min(left_r, cop_left_r)
    
    for i in range(n_cops):
        cop_row, cop_col = cops[i]
        cop_x, cop_y = xy_coordinate(cop_row + 0.5, cop_col + 0.5, n_rows)

        cop_left, cop_right = (orientations[i] + fov[i]/2) % 360, (orientations[i] - fov[i]/2) % 360
        cop_left_r = math.radians(cop_left)
        cop_right_r = math.radians(cop_right)

        for row in range(n_rows):
            for col in range(n_cols):
                if row == cop_row and col == cop_col:
                    seen[row][col] = 1
                    continue
                if row == cop_row:
                    # left
                    if col < cop_col and overlap(row+1, col+1, row, col+1):
                        if row == theif[0] and col == theif[1]:
                            cops_visible.append(i+1)
                        seen[row][col] = 1
                    # right
                    elif col > cop_col and overlap(row, col, row+1, col):
                        if row == theif[0] and col == theif[1]:
                            cops_visible.append(i+1)
                        seen[row][col] = 1
                elif col == cop_col:
                    # top
                    if row < cop_row and overlap(row+1, col, row+1, col+1):
                        if row == theif[0] and col == theif[1]:
                            cops_visible.append(i+1)
                        seen[row][col] = 1
                    # bottom
                    elif row > cop_row and overlap(row, col+1, row, col):
                        if row == theif[0] and col == theif[1]:
                            cops_visible.append(i+1)
                        seen[row][col] = 1
                elif row < cop_row:
                    # top-left
                    if col < cop_col and overlap(row+1, col, row, col+1):
                        if row == theif[0] and col == theif[1]:
                            cops_visible.append(i+1)
                        seen[row][col] = 1
                    # top-right
                    elif col > cop_col and overlap(row, col, row+1, col+1):
                        if row == theif[0] and col == theif[1]:
                            cops_visible.append(i+1)
                        seen[row][col] = 1
                else:
                    # bottom-left
                    if col < cop_col and overlap(row+1, col+1, row, col):
                        if row == theif[0] and col == theif[1]:
                            cops_visible.append(i+1)
                        seen[row][col] = 1
                    # bottom-right
                    elif col > cop_col and overlap(row, col+1, row+1, col):
                        if row == theif[0] and col == theif[1]:
                            cops_visible.append(i+1)
                        seen[row][col] = 1
    
    distance = n_rows + n_cols
    safe_pos = None
    for i in range(n_rows):
        for j in range(n_cols):
            if seen[i][j] == 0 and abs(i - theif[0]) + abs(j - theif[1]) < distance:
                distance = abs(i - theif[0]) + abs(j - theif[1])
                safe_pos = [i, j]

    return cops_visible, safe_pos

# Transform row-col coordinates to x-y coordinates with bottom-left origin
def xy_coordinate(row, col, n_rows):
    return col, n_rows - row


grid = [
    [0, 0, 0, 0, 0],
    ['T', 0, 0, 0, 2],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
]
orientations = [180,150]
fov = [60,60]

print(thief_and_cops(grid, orientations, fov))

grid = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 'T', 0],
    [0, 0, 0, 0]
]
orientations = [0]
fov = [30]
print(thief_and_cops(grid, orientations, fov))