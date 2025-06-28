// Aleksa Djukic E2 84/2024

use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    num::NonZero,
    process::exit,
    sync::{
        Arc, Condvar, Mutex, RwLock,
        atomic::{AtomicI32, Ordering},
        mpsc::channel,
    },
    thread::{self, available_parallelism},
    time::Instant,
    usize,
};

use grid::Grid;

fn main() {
    match load_data("input.txt") {
        Ok((distances, cities)) => {
            // pretty_print_grid(&distances);
            let now = Instant::now();
            let (min_distance, path) = held_karp_seq(&distances, distances.rows(), &cities);
            println!("---------- SEQUENTIAL ----------");
            println!("Min distance: {}", min_distance);
            println!("Path: {}", path);
            println!("Time: {:?}", now.elapsed());

            let now = Instant::now();
            let (min_distance, path) = held_karp_par(&distances, distances.rows(), &cities);
            println!("---------- PARALLEL ----------");
            println!("Min distance: {}", min_distance);
            println!("Path: {}", path);
            println!("Time: {:?}", now.elapsed());
        }
        Err(e) => {
            eprintln!("{e}");
            exit(1);
        }
    }
}

fn pretty_print_grid(distances: &Grid<usize>) {
    print!("\t");
    for i in 0..distances.rows() {
        print!("{i}\t");
    }
    println!();
    for (i, row) in distances.iter_rows().enumerate() {
        print!("{i}\t");
        for col in row {
            print!("{col}\t");
        }
        println!();
    }
}

fn load_data(filename: &str) -> Result<(Grid<usize>, HashMap<usize, String>), String> {
    let mut file = File::open(filename).unwrap();
    let mut data = String::new();

    if let Err(e) = file.read_to_string(&mut data) {
        return Err(e.to_string());
    }

    let rows = data.lines().count();
    let cities = (1 + (1 + 8 * rows).isqrt()) / 2;
    let mut distances: Grid<usize> = Grid::init(cities, cities, 0);
    let mut cities: HashMap<String, usize> = HashMap::new();
    let mut city_num = 0;
    for line in data.lines() {
        let line_parts = line.split(",").collect::<Vec<_>>();

        let mut indices = [0; 2];
        for i in (0..3).step_by(2) {
            indices[i / 2] = *cities
                .entry(format!("{},{}", line_parts[i], line_parts[i + 1]))
                .or_insert_with(|| {
                    city_num += 1;
                    city_num - 1
                });
        }

        if let Ok(distance) = line_parts[4].parse() {
            distances[(indices[0], indices[1])] = distance;
            distances[(indices[1], indices[0])] = distance;
        } else {
            return Err(format!("File {filename} has invalid data."));
        }
    }

    Ok((
        distances,
        cities.into_iter().map(|(key, val)| (val, key)).collect(),
    ))
}

fn get_combinations(size: usize, max_size: usize) -> Vec<usize> {
    let mut result = vec![];

    let mut bitset: usize = (1 << size) - 1;

    while bitset < (1 << max_size) {
        result.push(bitset);

        // Gosper's Hack
        let c = bitset & (!bitset + 1);
        let r = bitset + c;
        bitset = (((r ^ bitset) >> 2) / c) | r;
    }

    result
}

fn held_karp_seq(
    distances: &Grid<usize>,
    n: usize,
    cities: &HashMap<usize, String>,
) -> (usize, String) {
    let mut shortest_paths: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for i in 1..n {
        shortest_paths.insert((1 << i - 1, i), (distances[(0, i)], 0));
    }

    for subset_size in 2..n {
        for subset in get_combinations(subset_size, n - 1) {
            evaluate_subset_seq(distances, &mut shortest_paths, subset);
        }
    }

    let (min_cost, path) = find_best_cost_path(distances, n, cities, shortest_paths);

    (min_cost, path)
}

fn held_karp_par(
    distances: &Grid<usize>,
    n: usize,
    cities: &HashMap<usize, String>,
) -> (usize, String) {
    let mut shortest_paths: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for i in 1..n {
        shortest_paths.insert((1 << i - 1, i), (distances[(0, i)], 0));
    }

    let num_threads = available_parallelism()
        .unwrap_or(NonZero::new(4).unwrap())
        .get();

    let shortest_paths_arc = Arc::new(RwLock::new(shortest_paths));
    let distances_arc = Arc::new(distances.clone());

    let job_counter = Arc::new(AtomicI32::new(0));
    let pair = Arc::new((Mutex::new(()), Condvar::new()));

    let threads = (0..num_threads)
        .map(|i| {
            let distances_clone = distances_arc.clone();
            let shortest_paths_clone = shortest_paths_arc.clone();
            let (sender, receiver) = channel();
            let job_counter = job_counter.clone();
            let pair = pair.clone();

            (
                thread::spawn(move || {
                    while let Ok(subset) = receiver.recv() {
                        // println!("Thread {i} received {subset:#019b}");
                        evaluate_subset_par(&distances_clone, &shortest_paths_clone, subset);
                        // println!("Thread {i} finished job");
                        let x = job_counter.fetch_sub(1, Ordering::Release);
                        // println!("Thread {i} decremented {x}");
                        // if job_counter.fetch_sub(1, Ordering::Release) == 1 {
                        if x == 1 {
                            let _guard = pair.0.lock().unwrap();
                            // println!("Thread {i} notifies main thread");
                            pair.1.notify_one();
                        }
                    }
                }),
                sender,
            )
        })
        .collect::<Vec<_>>();

    let (lock, cvar) = &*pair;

    let mut channel_num = 0;
    for subset_size in 2..n {
        let mut count = 0;
        // println!("Main thread starts generating jobs");
        for subset in get_combinations(subset_size, n - 1) {
            threads[channel_num].1.send(subset).unwrap();
            channel_num = (channel_num + 1) % num_threads;
            count += 1;
        }

        // println!("Main thread added {count} jobs");
        if job_counter.fetch_add(count, Ordering::Acquire) != -count {
            let mut guard = lock.lock().unwrap();
            // println!("Waiting for threads to finish");
            guard = cvar.wait(guard).unwrap();
        }
    }

    for thread in threads {
        drop(thread.1);
        thread.0.join().unwrap();
    }

    let shortest_paths = RwLock::into_inner(Arc::into_inner(shortest_paths_arc).unwrap()).unwrap();
    let (min_cost, path) = find_best_cost_path(distances, n, cities, shortest_paths);

    (min_cost, path)
}

fn find_best_cost_path(
    distances: &Grid<usize>,
    n: usize,
    cities: &HashMap<usize, String>,
    shortest_paths: HashMap<(usize, usize), (usize, usize)>,
) -> (usize, String) {
    let mut min_cost = usize::MAX;
    let mut subset = 2_usize.pow(n as u32 - 1) - 1;
    let mut path = vec![0, 0];
    for i in 1..n {
        let cost = shortest_paths[&(subset, i)].0 + distances[(i, 0)];
        if min_cost > cost {
            min_cost = cost;
            if let Some(el) = path.last_mut() {
                *el = i;
            }
        }
    }

    while let Some(last_el) = path.last().cloned() {
        if last_el == 0 {
            break;
        }

        path.push(shortest_paths[&(subset, last_el)].1);
        subset = subset ^ 2_usize.pow(last_el as u32 - 1);
    }

    let path_str = path
        .iter()
        .map(|num| cities[num].clone())
        .rev()
        .collect::<Vec<_>>()
        .join(" -> ");
    (min_cost, path_str)
}

fn evaluate_subset_seq(
    distances: &Grid<usize>,
    shortest_paths: &mut HashMap<(usize, usize), (usize, usize)>,
    subset: usize,
) {
    let mut dest = 1;
    while dest < subset {
        if dest & subset != 0 {
            find_min_cost_path_via_subset(distances, shortest_paths, subset, dest);
        }
        dest <<= 1;
    }
}

fn evaluate_subset_par(
    distances: &Arc<Grid<usize>>,
    shortest_paths: &Arc<RwLock<HashMap<(usize, usize), (usize, usize)>>>,
    subset: usize,
) {
    let mut dest = 1;
    while dest < subset {
        if dest & subset != 0 {
            find_min_cost_path_via_subset_par(&distances, &shortest_paths, subset, dest);
        }
        dest <<= 1;
    }
}

fn find_min_cost_path_via_subset(
    distances: &Grid<usize>,
    shortest_paths: &mut HashMap<(usize, usize), (usize, usize)>,
    subset: usize,
    dest: usize,
) {
    let dest_idx = dest.ilog2() as usize + 1;
    let intermediate_nodes = subset ^ dest;
    let mut before_dest = 1;
    let mut min_cost = usize::MAX;
    let mut best_before_dest = before_dest;
    while before_dest <= intermediate_nodes {
        if before_dest & intermediate_nodes != 0 {
            let before_dest_idx = before_dest.ilog2() as usize + 1;
            let cost = shortest_paths[&(intermediate_nodes, before_dest_idx)].0
                + distances[(before_dest_idx, dest_idx)];
            if min_cost > cost {
                min_cost = cost;
                best_before_dest = before_dest_idx;
            }
        }
        before_dest <<= 1;
    }
    shortest_paths.insert((subset, dest_idx), (min_cost, best_before_dest));
}

fn find_min_cost_path_via_subset_par(
    distances: &Arc<Grid<usize>>,
    shortest_paths: &Arc<RwLock<HashMap<(usize, usize), (usize, usize)>>>,
    subset: usize,
    dest: usize,
) {
    let dest_idx = dest.ilog2() as usize + 1;
    let intermediate_nodes = subset ^ dest;
    let mut before_dest = 1;
    let mut min_cost = usize::MAX;
    let mut best_before_dest = before_dest;
    while before_dest <= intermediate_nodes {
        if before_dest & intermediate_nodes != 0 {
            let before_dest_idx = before_dest.ilog2() as usize + 1;
            let shortest_paths_read = shortest_paths.read().unwrap();
            let cost = shortest_paths_read[&(intermediate_nodes, before_dest_idx)].0
                + distances[(before_dest_idx, dest_idx)];
            if min_cost > cost {
                min_cost = cost;
                best_before_dest = before_dest_idx;
            }
        }
        before_dest <<= 1;
    }
    let mut shortest_paths_write = shortest_paths.write().unwrap();
    shortest_paths_write.insert((subset, dest_idx), (min_cost, best_before_dest));
}

#[test]
fn test_seq() {
    match load_data("input_test.txt") {
        Ok((distances, cities)) => {
            let (min_distance, path) = held_karp_seq(&distances, distances.rows(), &cities);
            assert_eq!(3243, min_distance);
            assert_eq!(
                "Paris,France -> Budapest,Hungary -> Vienna,Austria -> Prague,Czech Republic -> Berlin,Germany -> Paris,France",
                path
            );
        }
        Err(e) => {
            eprintln!("{e}");
            exit(1);
        }
    }
}

#[test]
fn test_par() {
    match load_data("input_test.txt") {
        Ok((distances, cities)) => {
            let (min_distance, path) = held_karp_par(&distances, distances.rows(), &cities);
            assert_eq!(3243, min_distance);
            assert_eq!(
                "Paris,France -> Budapest,Hungary -> Vienna,Austria -> Prague,Czech Republic -> Berlin,Germany -> Paris,France",
                path
            );
        }
        Err(e) => {
            eprintln!("{e}");
            exit(1);
        }
    }
}
