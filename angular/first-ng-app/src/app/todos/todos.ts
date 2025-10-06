import { Component, inject, OnInit, signal } from '@angular/core';
import { TodosService } from '../services/todos';
import { Todo } from '../model/todo.type';
import { catchError } from 'rxjs/internal/operators/catchError';

@Component({
  selector: 'app-todos',
  imports: [],
  templateUrl: './todos.html',
  styleUrl: './todos.scss'
})

/* OnInit which is a lifecycle Loop
this allows you to run a function when
initiated ngOnInit() */
export class Todos implements OnInit {
  todosService = inject(TodosService);
  items = signal<Array<Todo>>([]);

  ngOnInit(): void {
    // console.log(this.todosService.todoItems);
    this.todosService.getTodosFromApi()
      .pipe(
        catchError(err =>
        {
          console.log(err);
          throw err;
        }
        )
      )
      .subscribe(todos => 
        this.items.set(todos)
      )
      
  }
}
