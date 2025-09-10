import { TestBed } from '@angular/core/testing';

import { TodosService } from './todos';

describe('Todos', () => {
  let service: TodosService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(TodosService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
